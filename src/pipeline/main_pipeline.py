"""
Main pipeline for generating product images with new backgrounds.

Orchestrates segmentation, quality checks, background generation/loading, 
and final image composition.
"""

from pathlib import Path
from typing import Union, Optional, Dict, Any, Tuple
from PIL import Image, ImageFilter
import numpy as np
import time
import cv2
import os
import logging # Added
import sys # Added for logging setup

# Import refactored components using absolute paths from src
from src.image.segmentation import Segmenter # Use Segmenter class
from src.image.quality import ImageAssessor
from src.background.generators import generate_standard_backgrounds, create_gradient_background, create_solid_background, generate_solid_background, generate_gradient_background # For synthetic backgrounds
from src.background.utils import load_background_image, combine_foreground_background, add_simple_drop_shadow # Note: add_simple_drop_shadow might be unused now
from src.models.diffusion import DiffusionGenerator # If using diffusion
from src.utils.data_io import save_image, load_image
# Use absolute import from src root for config
from src.config import load_config, DEFAULT_CONFIG # Import config loading function and defaults
from src.image import filtering # Added import

# --- Helper Function for Improved Shadow ---

def add_soft_drop_shadow(
    image_rgba: np.ndarray,
    offset: Tuple[int, int] = (5, 5),
    blur_sigma: float = 5.0,
    opacity: float = 0.5,
    shadow_color: Tuple[int, int, int] = (0, 0, 0)
) -> np.ndarray:
    """
    Adds a softer, configurable drop shadow to an RGBA image.

    Args:
        image_rgba (np.ndarray): Input RGBA image (H, W, 4) with the foreground object.
        offset (Tuple[int, int]): (x_offset, y_offset) for the shadow.
        blur_sigma (float): Gaussian blur sigma for shadow softness. If <= 0, no blur.
        opacity (float): Opacity of the shadow (0.0 to 1.0).
        shadow_color (Tuple[int, int, int]): RGB color of the shadow.

    Returns:
        np.ndarray: The image (H, W, 4) with the shadow added behind the foreground.
    """
    if image_rgba.shape[2] != 4:
        raise ValueError("Input image must be RGBA (H, W, 4)")

    alpha = image_rgba[:, :, 3] # Extract alpha channel (H, W)
    h, w = alpha.shape

    # Create shadow mask (same shape as alpha, initially black where object is opaque)
    shadow_mask = (alpha > 10).astype(np.uint8) * 255 # Threshold alpha slightly

    # Apply Gaussian blur for softness
    if blur_sigma > 0:
        # Kernel size must be odd, choose based on sigma
        k_size = int(6 * blur_sigma + 1) 
        if k_size % 2 == 0: k_size += 1 # Ensure odd
        shadow_mask = cv2.GaussianBlur(shadow_mask, (k_size, k_size), blur_sigma)

    # Create RGBA shadow layer
    shadow_layer = np.zeros((h, w, 4), dtype=np.uint8)
    shadow_layer[:, :, 0] = shadow_color[0]
    shadow_layer[:, :, 1] = shadow_color[1]
    shadow_layer[:, :, 2] = shadow_color[2]
    # Apply blurred mask as alpha, scaled by opacity
    shadow_layer[:, :, 3] = (shadow_mask * opacity).astype(np.uint8)

    # Shift the shadow layer
    x_off, y_off = offset
    M = np.float32([[1, 0, x_off], [0, 1, y_off]])
    shifted_shadow = cv2.warpAffine(shadow_layer, M, (w, h))

    # Composite shadow behind original image using alpha blending
    # Create a PIL image for easier compositing
    shifted_shadow_pil = Image.fromarray(shifted_shadow)
    original_pil = Image.fromarray(image_rgba)

    # Create a black background to composite shadow onto first
    composite_base = Image.new('RGBA', original_pil.size, (0, 0, 0, 0))
    # Paste shadow onto base
    composite_base.paste(shifted_shadow_pil, (0, 0), shifted_shadow_pil)
    # Paste original foreground over the shadow
    composite_base.paste(original_pil, (0, 0), original_pil)

    return np.array(composite_base)

# --- Pipeline Class ---

class GenerationPipeline:
    """
    Handles the end-to-end process of background replacement for a product image.
    """

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        # Allow overrides via arguments, these take precedence over config file
        segmenter_model: Optional[str] = None,
        diffusion_enabled: Optional[bool] = None,
        diffusion_cfg_overrides: Optional[Dict[str, Any]] = None,
        save_intermediate_masks_override: Optional[bool] = None,
    ):
        """
        Initializes the pipeline with necessary components, loading configuration.
        Includes overrides for key parameters.
        """
        # --- Basic Logging Setup --- 
        # Ideally, setup logging in the main entry point (main.py or generate.py)
        # For now, setting up basic config here if not already configured.
        if not logging.getLogger().hasHandlers():
            logging.basicConfig(
                level=logging.INFO, 
                format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                stream=sys.stdout # Log to stdout
            )
        # ---------------------------    
            
        # Load base configuration (defaults or from file)
        self.config = load_config(config_path) # load_config returns a copy

        # Apply direct overrides from arguments to the config dictionary
        if segmenter_model is not None:
            self.config['segmenter_model'] = segmenter_model
        if diffusion_enabled is not None:
            self.config['diffusion_enabled'] = diffusion_enabled # Override top-level key
        if save_intermediate_masks_override is not None:
            self.config['save_intermediate_masks'] = save_intermediate_masks_override

        logging.info("Initializing pipeline components...") # Use logging

        # Initialize Segmenter using the potentially overridden value
        self.segmenter = Segmenter(model_name=self.config.get('segmenter_model', 'u2net'))
        logging.info(f"Segmenter initialized with model: {self.segmenter.model_name}") # Use logging

        # Initialize ImageAssessor (assuming simplified usage or removal)
        # qa_cfg = self.config.get('quality_assessment', {})
        self.assessor = ImageAssessor(
             min_resolution=self.config.get('min_resolution', (300, 300)),
             blur_threshold=self.config.get('blur_threshold', 100.0),
             contrast_threshold=self.config.get('contrast_threshold', 30.0),
             # bg_complexity_threshold=qa_cfg.get('bg_complexity_threshold', 0.1)
        )
        logging.info("ImageAssessor initialized.") # Use logging

        # Initialize diffusion only if enabled in the config
        self.diffusion_generator: Optional[DiffusionGenerator] = None
        # Read the potentially overridden diffusion_enabled flag from self.config
        is_diffusion_enabled = self.config.get('diffusion_enabled', False)

        if is_diffusion_enabled:
             logging.info("Diffusion is enabled. Initializing Diffusion Generator...") # Use logging
             # Build the config for DiffusionGenerator directly from top-level keys
             final_diff_cfg = {
                # Use .get() with fallbacks to defaults from DEFAULT_CONFIG dict itself
                # Map config keys to DiffusionGenerator __init__ args
                'sd_model_id': self.config.get('diffusion_model_id', DEFAULT_CONFIG.get('diffusion_model_id')),
                # Use 'custom_controlnet_id' which DiffusionGenerator expects
                'custom_controlnet_id': self.config.get('diffusion_controlnet_model_id', DEFAULT_CONFIG.get('diffusion_controlnet_model_id')),
                # Add controlnet_type, defaulting to 'seg'
                'controlnet_type': self.config.get('diffusion_controlnet_type', DEFAULT_CONFIG.get('diffusion_controlnet_type', 'seg')),
                'device': self.config.get('diffusion_device', DEFAULT_CONFIG.get('diffusion_device')),
                'scheduler_type': self.config.get('diffusion_scheduler', DEFAULT_CONFIG.get('diffusion_scheduler')),
                'enable_cpu_offload': self.config.get('diffusion_enable_cpu_offload', DEFAULT_CONFIG.get('diffusion_enable_cpu_offload'))
                # Add other DiffusionGenerator params here if needed, fetching from self.config
             }

             # Apply specific JSON overrides if provided, but remove num_inference_steps from initialization
             if diffusion_cfg_overrides:
                  logging.info(f"Applying diffusion overrides: {diffusion_cfg_overrides}") # Use logging
                  # Create a copy of overrides without num_inference_steps for initialization
                  init_overrides = {k: v for k, v in diffusion_cfg_overrides.items() if k != 'num_inference_steps'}
                  final_diff_cfg.update(init_overrides)
                  # Store num_inference_steps separately for use during generation
                  self.diffusion_num_inference_steps = diffusion_cfg_overrides.get('num_inference_steps')
             else:
                  self.diffusion_num_inference_steps = None

             # Filter out None values as DiffusionGenerator might expect specific types
             final_diff_cfg_filtered = {k: v for k, v in final_diff_cfg.items() if v is not None}

             try:
                 logging.info(f"Initializing Diffusion Generator with effective config: {final_diff_cfg_filtered}") # Use logging
                 self.diffusion_generator = DiffusionGenerator(**final_diff_cfg_filtered)
                 logging.info("Diffusion Generator initialized successfully.") # Use logging
             except Exception as e:
                 # Error is logged within DiffusionGenerator now, just log warning here
                 logging.warning(f"Pipeline: Failed to initialize DiffusionGenerator. Diffusion will be unavailable. Error: {e}") # Use logging
                 self.diffusion_generator = None
        else:
             logging.info("Diffusion is disabled.") # Use logging

        logging.info("Pipeline initialized.") # Use logging

    def process_image(
        self,
        image_path: Union[str, Path],
        output_path: Union[str, Path],
        background_spec: Union[str, Path, Tuple, Dict[str, Any]],
        prompt: Optional[str] = None
    ) -> bool:
        """
        Processes a single image: load, assess, segment, filter/refine mask, feather mask,
        generate/load background, add shadow, combine.
        """
        start_time = time.time()
        image_path = Path(image_path)
        output_path = Path(output_path)
        output_basename = output_path.stem
        logging.info(f"--- Starting processing for: {image_path.name} -> {output_path.name} ---")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # --- Configurable Flags/Dirs/Params ---
        save_intermediate = self.config.get('save_intermediate_masks', False)
        intermediate_dir = Path(self.config.get('intermediate_mask_dir')) if save_intermediate else None
        if intermediate_dir:
             intermediate_dir.mkdir(parents=True, exist_ok=True)
             
        refine_mask_flag = self.config.get('refine_mask', True)
        feather_amount = self.config.get('edge_feathering_amount', 0.0)
        add_shadow_flag = self.config.get('add_shadow', True)

        # --- Load Image ---
        logging.info("1. Loading image...")
        input_pil = load_image(image_path, mode='RGB')
        if input_pil is None: return False
        W, H = input_pil.size
        input_rgb_np = np.array(input_pil)

        # --- Assess Quality --- (Currently skipped)
        logging.info("2. Assessing image quality...")
        logging.info("   Skipping detailed quality assessment.")

        # --- Segment Foreground --- 
        logging.info("3. Segmenting foreground...")
        raw_mask_np: Optional[np.ndarray] = None
        try:
            # Segmenter now only returns the raw mask
            raw_mask_np = self.segmenter.segment(
                input_rgb_np, # Pass numpy array
                save_intermediate=save_intermediate,
                intermediate_dir=intermediate_dir,
                output_basename=output_basename
            )
            if raw_mask_np is None: 
                # Error should be logged in segmenter
                raise RuntimeError("Segmentation returned None.")
            logging.info("   Raw segmentation complete.")
        except Exception as e:
            logging.error(f"Error during segmentation step: {e}", exc_info=True)
            return False
        
        # --- Filtering & Refinement Stage --- 
        logging.info("4. Filtering and Refining Mask...")
        final_mask_np = raw_mask_np # Start with raw mask
        filter_passed = True
        filter_reason = ""
        
        # --- Apply Configured Filters Sequentially ---
        active_filters = []
        
        if self.config.get('apply_contour_filter', False):
            active_filters.append(("Contour", filtering.filter_by_contour))

        if self.config.get('apply_contrast_filter', False):
            active_filters.append(("Contrast", filtering.filter_by_contrast))

        if self.config.get('apply_text_filter', False):
             # Implementation skipped, but keep check if flag might be true
             logging.debug("Text filter enabled in config but implementation skipped.")
             # active_filters.append(("Text", filtering.filter_by_text_overlap))
             
        if self.config.get('apply_clutter_filter', False):
             active_filters.append(("Clutter", filtering.filter_by_clutter))
        # ----------------------------------------------
        
        logging.info(f"Active mask filters: {[name for name, _ in active_filters]}")

        for filter_name, filter_func in active_filters:
            try:
                logging.debug(f"Applying {filter_name} filter...")
                filter_passed, filter_reason = filter_func(input_rgb_np, final_mask_np, self.config)
                if not filter_passed:
                    logging.warning(f"Mask rejected by {filter_name} filter: {filter_reason}")
                    break # Stop filtering on first failure
                else:
                    logging.debug(f"{filter_name} filter passed.")
            except Exception as filter_e:
                logging.error(f"Error during {filter_name} filtering: {filter_e}", exc_info=True)
                filter_passed = False
                filter_reason = f"Error in {filter_name} filter"
                break

        if not filter_passed:
            logging.warning(f"Skipping image {image_path.name} due to failed mask filter ({filter_reason}).")
            return False # Image rejected by filters
        else:
            logging.info("   Mask passed all active filters.")

        # Apply Morphological Refinement (if enabled and filters passed)
        if refine_mask_flag:
            logging.info("   Applying morphological refinement...")
            try:
                final_mask_np = filtering.refine_morphological(final_mask_np, self.config)
                logging.info("   Morphological refinement complete.")
                # Save refined mask if requested (moved from Segmenter)
                if save_intermediate and intermediate_dir:
                    try:
                        save_path = intermediate_dir / f"{output_basename}_mask_refined.png"
                        save_image(Image.fromarray(final_mask_np), save_path)
                        logging.info(f"Saved refined mask to {save_path}")
                    except Exception as save_e:
                        logging.error(f"Failed to save refined mask to {save_path}: {save_e}")
            except Exception as refine_e:
                logging.error(f"Error during morphological refinement: {refine_e}", exc_info=True)
                return False # Treat refinement error as failure for now
        else:
            logging.info("   Skipping morphological refinement (disabled in config).")

        # --- Feather Mask Edges --- (Now Step 5)
        logging.info("5. Feathering mask edges (optional)...")
        if feather_amount > 0:
            # ... (existing feathering logic using final_mask_np) ...
            logging.info(f"   Feathering mask edges (sigma={feather_amount})...")
            mask_float = final_mask_np.astype(np.float32) / 255.0
            k_size = int(6 * feather_amount + 1)
            if k_size % 2 == 0: k_size += 1
            try:
                feathered_mask_float = cv2.GaussianBlur(mask_float, (k_size, k_size), feather_amount)
                final_mask_np = (feathered_mask_float * 255).clip(0, 255).astype(np.uint8)
                logging.info("   Mask feathering complete.")
            except Exception as e:
                 logging.warning(f"Failed to feather mask: {e}. Using unfeathered mask.", exc_info=True)
                 # Fallback handled by keeping final_mask_np as is
        else:
             logging.info("   Skipping mask edge feathering.")

        # Convert final mask to PIL for diffusion generator 
        final_mask_pil = Image.fromarray(final_mask_np).convert('L')

        # Create RGBA foreground (using final mask)
        foreground_rgba_np = cv2.cvtColor(input_rgb_np, cv2.COLOR_RGB2RGBA)
        foreground_rgba_np[:, :, 3] = final_mask_np

        # --- Prepare Background --- (Now Step 6)
        logging.info(f"6. Preparing background... Spec: {background_spec}")
        background_rgb_np: Optional[np.ndarray] = None
        target_size = (W, H)
        bg_type = 'unknown'

        try:
            # Logic for handling background_spec (file, tuple, dict)
            if isinstance(background_spec, (str, Path)):
                bg_type = 'file'
                logging.info(f"   Type: File ({background_spec})") # Use logging
                background_rgb_np = load_background_image(background_spec, target_size)
                if background_rgb_np is None: raise ValueError(f"Failed loading bg: {background_spec}")
            elif isinstance(background_spec, tuple) and len(background_spec) == 3:
                 bg_type = 'solid'
                 logging.info(f"   Type: Solid Color ({background_spec})") # Use logging
                 bg_pil = generate_solid_background(width=W, height=H, color=background_spec)
                 if bg_pil: background_rgb_np = np.array(bg_pil)
                 else: raise ValueError(f"Failed generating solid bg: {background_spec}")
            elif isinstance(background_spec, dict):
                 bg_type = background_spec.get('type', 'unknown')
                 logging.info(f"   Type: Dict ({bg_type})") # Use logging
                 if bg_type == 'file':
                     bg_path = background_spec.get('path')
                     if not bg_path: raise ValueError("Bg type 'file' needs 'path'.")
                     background_rgb_np = load_background_image(bg_path, target_size)
                     if background_rgb_np is None: raise ValueError(f"Failed loading bg: {bg_path}")
                 elif bg_type == 'solid':
                     color = background_spec.get('color', self.config.get('default_bg_color', (240,240,240)))
                     bg_pil = generate_solid_background(width=W, height=H, color=color)
                     if bg_pil: background_rgb_np = np.array(bg_pil)
                     else: raise ValueError(f"Failed generating solid bg: {color}")
                 elif bg_type == 'gradient':
                     colors = background_spec.get('colors', self.config.get('default_gradient_colors', [(245,245,245),(230,230,230)]))
                     direction = background_spec.get('direction', 'vertical')
                     bg_pil = generate_gradient_background(width=W, height=H, colors=colors, direction=direction)
                     if bg_pil: background_rgb_np = np.array(bg_pil)
                     else: raise ValueError("Failed generating gradient bg.")
                 elif bg_type == 'diffusion':
                     if not self.diffusion_generator:
                          # Log error instead of raising immediately?
                          logging.error("Diffusion requested but generator not initialized.") # Use logging
                          raise RuntimeError("Diffusion requested but generator not initialized.") # Keep raise for now
                     effective_prompt = background_spec.get('prompt', prompt)
                     if not effective_prompt: 
                          logging.error("Diffusion requested but no prompt provided.") # Use logging
                          raise ValueError("Diffusion requested but no prompt provided.")
                     logging.info(f"   Generating diffusion background with prompt: '{effective_prompt}'") # Use logging
                     
                     # Get diffusion parameters from config
                     diff_params = {
                         'num_inference_steps': self.diffusion_num_inference_steps or self.config.get('diffusion_num_inference_steps', 30),
                         'guidance_scale': self.config.get('diffusion_guidance_scale', 7.5),
                         'controlnet_conditioning_scale': self.config.get('diffusion_controlnet_scale', 0.75)
                     }
                     logging.info(f"   Diffusion params: {diff_params}") # Use logging

                     # Get target processing resolution from config
                     target_res = self.config.get('diffusion_processing_resolution')
                     if target_res and not (isinstance(target_res, (tuple, list)) and len(target_res) == 2):
                         logging.warning(f"Invalid diffusion_processing_resolution in config: {target_res}. Must be tuple/list of 2 ints. Using default resizing.")
                         target_res = None
                     else:
                         logging.info(f"   Using target diffusion processing resolution: {target_res}")

                     # DiffusionGenerator might need PIL image and PIL mask
                     bg_pil = self.diffusion_generator.generate(
                         image_input=input_pil, # Original RGB PIL
                         foreground_mask=final_mask_pil, # Final (feathered) L mask PIL
                         prompt=effective_prompt,
                         target_processing_size=target_res, # Pass target size
                         **diff_params
                     )
                     if bg_pil is None: 
                         # Error already logged in diffusion.py
                         logging.error("Pipeline: Diffusion generation failed.") # Use logging
                         raise RuntimeError("Diffusion generation failed.") # Keep raise
                     background_rgb_np = np.array(bg_pil.convert('RGB'))
                 else:
                     raise ValueError(f"Unknown background type in dict: {bg_type}")
            else:
                 raise TypeError(f"Invalid background_spec type: {type(background_spec)}")

            if background_rgb_np is None:
                 raise ValueError("Background preparation resulted in None.")

            logging.info("   Background prepared.") # Use logging
        except Exception as e:
            logging.error(f"Error preparing background: {e}", exc_info=True) # Use logging
            return False

        # --- Add Shadow --- (Now Step 7)
        logging.info("7. Adding drop shadow (optional)...")
        foreground_with_shadow_np = foreground_rgba_np
        if add_shadow_flag:
            logging.info("6. Adding drop shadow...") # Use logging
            shadow_params = {
                'offset': (
                    self.config.get('shadow_offset_x', 5),
                    self.config.get('shadow_offset_y', 5)
                ),
                'blur_sigma': self.config.get('shadow_blur_sigma', 5.0),
                'opacity': self.config.get('shadow_opacity', 0.5),
                'shadow_color': self.config.get('shadow_color', (0, 0, 0))
            }
            logging.info(f"   Shadow params: {shadow_params}") # Use logging
            try:
                foreground_with_shadow_np = add_soft_drop_shadow(foreground_rgba_np, **shadow_params)
                logging.info("   Shadow added.") # Use logging
            except Exception as e:
                logging.warning(f"Failed to add shadow: {e}. Proceeding without shadow.", exc_info=True) # Use logging
                foreground_with_shadow_np = foreground_rgba_np
        else:
            logging.info("6. Skipping drop shadow.") # Use logging

        # --- Combine Layers --- (Now Step 8)
        logging.info("8. Combining final layers...")
        try:
            # Ensure background is also RGBA for consistent compositing
            if background_rgb_np.shape[2] == 3:
                background_rgba_np = cv2.cvtColor(background_rgb_np, cv2.COLOR_RGB2RGBA)
                background_rgba_np[:, :, 3] = 255 # Make background fully opaque
            else:
                background_rgba_np = background_rgb_np # Assume it's already RGBA if not 3 channels

            # Convert final foreground (with shadow) and background to PIL for pasting
            foreground_pil = Image.fromarray(foreground_with_shadow_np)
            background_pil = Image.fromarray(background_rgba_np)

            # Simple alpha compositing: Paste foreground onto background using foreground's alpha
            # This works because add_soft_drop_shadow returns an image where the shadow
            # is already blended into the transparent areas behind the object.
            final_image_pil = background_pil.copy()
            final_image_pil.paste(foreground_pil, (0, 0), foreground_pil) # Paste using fg alpha
            logging.info("   Layers combined.") # Use logging
        except Exception as e:
            logging.error(f"Error combining layers: {e}", exc_info=True) # Use logging
            return False

        # --- Save Result --- (Now Step 9)
        logging.info(f"9. Saving final image to: {output_path}")
        # Convert to RGB before saving unless transparency is desired (e.g., PNG)
        save_mode = 'RGBA' if output_path.suffix.lower() == '.png' else 'RGB'
        success = save_image(final_image_pil.convert(save_mode), output_path)

        end_time = time.time()
        logging.info(f"--- Processing finished in {end_time - start_time:.2f} seconds. Success: {success} ---")
        return success

# --- Example Usage (for testing within the module) ---
# if __name__ == '__main__':
#     # Setup basic logging for the example
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
#     logging.info("Testing pipeline...")
#     # Setup paths relative to project root or use absolute paths
#     # Assumes running from project root where `data` and `results` exist
#     test_image = Path("../data/images/your_test_image.jpg") # CHANGE THIS
#     output_dir = Path("../results/pipeline_test")
#     output_dir.mkdir(parents=True, exist_ok=True)
# 
#     if not test_image.exists():
#         logging.error(f"Test image not found: {test_image}")
#     else:
#         # --- Initialize Pipeline ---
#         # Example with diffusion enabled (requires compatible hardware & models downloaded)
#         # diffusion_overrides = {
#         #     'sd_model_id': 'runwayml/stable-diffusion-v1-5',
#         #     'controlnet_type': 'canny', # or 'seg'
#         #     'device': 'cuda' # or 'cpu'
#         # }
#         # pipeline = GenerationPipeline(diffusion_enabled=True, diffusion_cfg_overrides=diffusion_overrides)
#         
#         # Example without diffusion (uses config defaults)
#         pipeline = GenerationPipeline() 
# 
#         # --- Test Cases ---
#         test_cases = [
#             # 1. Solid background
#             {'output': output_dir / f"{test_image.stem}_solid_white.png", 'bg': (255, 255, 255), 'prompt': None},
#             # 2. Gradient background
#             {'output': output_dir / f"{test_image.stem}_gradient_blue.png", 'bg': {'type': 'gradient', 'colors': [(230, 240, 255), (200, 210, 230)]}, 'prompt': None},
#             # 3. Background from file
#             # {'output': output_dir / f"{test_image.stem}_bg_file.png", 'bg': 'path/to/your/background.jpg', 'prompt': None},
#             # 4. Diffusion background (if diffusion_generator is initialized)
#             # {'output': output_dir / f"{test_image.stem}_diffusion_studio.png", 'bg': {'type': 'diffusion'}, 'prompt': 'Clean white photography studio background, soft lighting'},
#         ]
# 
#         for i, case in enumerate(test_cases):
#             logging.info(f"\n--- Running Test Case {i+1} ---")
#             # Skip diffusion if not available
#             if isinstance(case['bg'], dict) and case['bg']['type'] == 'diffusion' and not pipeline.diffusion_generator:
#                 logging.warning("Skipping diffusion test case as DiffusionGenerator is not available.")
#                 continue
#                 
#             pipeline.process_image(
#                 image_path=test_image,
#                 output_path=case['output'],
#                 background_spec=case['bg'],
#                 prompt=case['prompt']
#             ) 