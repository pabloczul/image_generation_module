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

# Import refactored components using absolute paths from src
from src.image.segmentation import Segmenter # Use Segmenter class
from src.image.quality import ImageAssessor
from src.background.generators import generate_standard_backgrounds, create_gradient_background, create_solid_background, generate_solid_background, generate_gradient_background # For synthetic backgrounds
from src.background.utils import load_background_image, combine_foreground_background, add_simple_drop_shadow # Note: add_simple_drop_shadow might be unused now
from src.models.diffusion import DiffusionGenerator # If using diffusion
from src.utils.data_io import save_image, load_image
# Use absolute import from src root for config
from src.config import load_config, DEFAULT_CONFIG # Import config loading function and defaults

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
        # Load base configuration (defaults or from file)
        self.config = load_config(config_path) # load_config returns a copy

        # Apply direct overrides from arguments to the config dictionary
        if segmenter_model is not None:
            self.config['segmenter_model'] = segmenter_model
        if diffusion_enabled is not None:
            self.config['diffusion_enabled'] = diffusion_enabled # Override top-level key
        if save_intermediate_masks_override is not None:
            self.config['save_intermediate_masks'] = save_intermediate_masks_override

        print("Initializing pipeline components...")

        # Initialize Segmenter using the potentially overridden value
        self.segmenter = Segmenter(model_name=self.config.get('segmenter_model', 'u2net'))
        print(f"Segmenter initialized with model: {self.segmenter.model_name}")

        # Initialize ImageAssessor (assuming simplified usage or removal)
        # qa_cfg = self.config.get('quality_assessment', {})
        self.assessor = ImageAssessor(
             min_resolution=self.config.get('min_resolution', (300, 300)),
             blur_threshold=self.config.get('blur_threshold', 100.0),
             contrast_threshold=self.config.get('contrast_threshold', 30.0),
             # bg_complexity_threshold=qa_cfg.get('bg_complexity_threshold', 0.1)
        )
        print("ImageAssessor initialized.")

        # Initialize diffusion only if enabled in the config
        self.diffusion_generator: Optional[DiffusionGenerator] = None
        # Read the potentially overridden diffusion_enabled flag from self.config
        is_diffusion_enabled = self.config.get('diffusion_enabled', False)

        if is_diffusion_enabled:
             print("Diffusion is enabled. Initializing Diffusion Generator...")
             # Build the config for DiffusionGenerator directly from top-level keys
             final_diff_cfg = {
                # Use .get() with fallbacks to defaults from DEFAULT_CONFIG dict itself
                # Map config keys to DiffusionGenerator __init__ args
                'sd_model_id': self.config.get('diffusion_model_id', DEFAULT_CONFIG['diffusion_model_id']),
                'controlnet_model_id': self.config.get('diffusion_controlnet_model_id', DEFAULT_CONFIG['diffusion_controlnet_model_id']),
                'device': self.config.get('diffusion_device', DEFAULT_CONFIG['diffusion_device']),
                'scheduler_type': self.config.get('diffusion_scheduler', DEFAULT_CONFIG['diffusion_scheduler']),
                'enable_cpu_offload': self.config.get('diffusion_enable_cpu_offload', DEFAULT_CONFIG['diffusion_enable_cpu_offload'])
                # Add other DiffusionGenerator params here if needed, fetching from self.config
             }

             # Apply specific JSON overrides if provided
             if diffusion_cfg_overrides:
                  print(f"Applying diffusion overrides: {diffusion_cfg_overrides}")
                  final_diff_cfg.update(diffusion_cfg_overrides)

             # Filter out None values as DiffusionGenerator might expect specific types
             final_diff_cfg_filtered = {k: v for k, v in final_diff_cfg.items() if v is not None}

             try:
                 print(f"Initializing Diffusion Generator with effective config: {final_diff_cfg_filtered}")
                 self.diffusion_generator = DiffusionGenerator(**final_diff_cfg_filtered)
                 print("Diffusion Generator initialized successfully.")
             except Exception as e:
                 print(f"Warning: Failed to initialize DiffusionGenerator: {e}. Diffusion background generation will not be available.")
                 # Keep self.diffusion_generator as None
                 self.diffusion_generator = None
        else:
             print("Diffusion is disabled.")

        print("Pipeline initialized.")

    def process_image(
        self,
        image_path: Union[str, Path],
        output_path: Union[str, Path],
        background_spec: Union[str, Path, Tuple, Dict[str, Any]],
        prompt: Optional[str] = None
    ) -> bool:
        """
        Processes a single image: load, assess, segment, refine mask, feather mask,
        generate/load background, add shadow, combine.

        Args:
            image_path: Path to the input product image.
            output_path: Path to save the resulting image.
            background_spec: Defines the background (see config comments).
            prompt (Optional[str]): Text prompt for diffusion.

        Returns:
            bool: True if successful, False otherwise.
        """
        start_time = time.time()
        image_path = Path(image_path)
        output_path = Path(output_path)
        # Use output filename (without extension) as base for intermediate files
        output_basename = output_path.stem
        print(f"--- Starting processing for: {image_path.name} -> {output_path.name} ---")

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # --- Configurable Flags/Dirs/Params ---
        save_intermediate = self.config.get('save_intermediate_masks', False)
        intermediate_dir = self.config.get('intermediate_mask_dir') if save_intermediate else None
        refine_mask_flag = self.config.get('refine_mask', True)
        feather_amount = self.config.get('edge_feathering_amount', 0.0)
        add_shadow_flag = self.config.get('add_shadow', True)

        # 1. Load Image
        print("1. Loading image...")
        input_pil = load_image(image_path, mode='RGB')
        if input_pil is None: return False # Error logged in load_image
        W, H = input_pil.size
        # Keep numpy version for CV operations
        input_rgb_np = np.array(input_pil)

        # 2. Assess Quality (Simplified/Placeholder)
        print("2. Assessing image quality...")
        # Assuming basic checks or skipping detailed assessment for now
        # quality_assessment = self.assessor.assess_quality(input_pil) # Original call
        # print(f"Quality Assessment: {quality_assessment['message']}")
        # if not quality_assessment['is_processable']:
        #      print("Warning: Proceeding despite low quality assessment.")
        print("   Skipping detailed quality assessment.")

        # 3. Segment Foreground
        print("3. Segmenting foreground...")
        try:
            # Pass flags and config directly to segmenter
            segmentation_params = {
                'refine': refine_mask_flag,
                'refinement_config': self.config, # Segmenter pulls needed keys
                'save_intermediate': save_intermediate,
                'intermediate_dir': intermediate_dir,
                'output_basename': output_basename,
            }
            # Segment now takes numpy array and returns final numpy mask (raw or refined)
            final_mask_np = self.segmenter.segment(input_rgb_np, return_rgba=False, **segmentation_params)
            if final_mask_np is None: raise RuntimeError("Segmentation returned None.")
            print("   Segmentation complete.")
        except Exception as e:
            print(f"Error during segmentation: {e}")
            return False

        # 4. Feather Mask Edges (Optional)
        if feather_amount > 0:
            print(f"4. Feathering mask edges (sigma={feather_amount})...")
            # Feathering needs float mask
            mask_float = final_mask_np.astype(np.float32) / 255.0
            # Kernel size must be odd
            k_size = int(6 * feather_amount + 1)
            if k_size % 2 == 0: k_size += 1
            try:
                feathered_mask_float = cv2.GaussianBlur(mask_float, (k_size, k_size), feather_amount)
                # Convert back to uint8 for compositing
                final_mask_np = (feathered_mask_float * 255).clip(0, 255).astype(np.uint8)
                print("   Mask feathering complete.")
            except Exception as e:
                 print(f"Warning: Failed to feather mask: {e}. Using unfeathered mask.")
                 # Fallback to the unfeathered mask if blur fails
                 final_mask_np = final_mask_np
        else:
             print("4. Skipping mask edge feathering.")

        # Convert final mask to PIL (needed for diffusion? Check DiffusionGenerator usage)
        final_mask_pil = Image.fromarray(final_mask_np).convert('L')

        # Create RGBA foreground numpy array using the final mask
        foreground_rgba_np = cv2.cvtColor(input_rgb_np, cv2.COLOR_RGB2RGBA)
        foreground_rgba_np[:, :, 3] = final_mask_np

        # 5. Prepare Background
        print(f"5. Preparing background... Spec: {background_spec}")
        background_rgb_np: Optional[np.ndarray] = None
        target_size = (W, H)
        bg_type = 'unknown'

        try:
            # Logic for handling background_spec (file, tuple, dict)
            if isinstance(background_spec, (str, Path)):
                bg_type = 'file'
                print(f"   Type: File ({background_spec})")
                background_rgb_np = load_background_image(background_spec, target_size)
                if background_rgb_np is None: raise ValueError(f"Failed loading bg: {background_spec}")
            elif isinstance(background_spec, tuple) and len(background_spec) == 3:
                 bg_type = 'solid'
                 print(f"   Type: Solid Color ({background_spec})")
                 bg_pil = generate_solid_background(width=W, height=H, color=background_spec)
                 if bg_pil: background_rgb_np = np.array(bg_pil)
                 else: raise ValueError(f"Failed generating solid bg: {background_spec}")
            elif isinstance(background_spec, dict):
                 bg_type = background_spec.get('type', 'unknown')
                 print(f"   Type: Dict ({bg_type})")
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
                          raise RuntimeError("Diffusion requested but generator not initialized.")
                     effective_prompt = background_spec.get('prompt', prompt)
                     if not effective_prompt: raise ValueError("Diffusion requested but no prompt provided.")
                     print(f"   Generating diffusion background with prompt: '{effective_prompt}'")
                     # Get diffusion parameters from config
                     diff_params = {
                         'num_inference_steps': self.config.get('diffusion_num_inference_steps', 30),
                         'guidance_scale': self.config.get('diffusion_guidance_scale', 7.5),
                         'controlnet_conditioning_scale': self.config.get('diffusion_controlnet_scale', 0.75)
                     }
                     print(f"   Diffusion params: {diff_params}")
                     # DiffusionGenerator might need PIL image and PIL mask
                     bg_pil = self.diffusion_generator.generate(
                         image_input=input_pil, # Original RGB PIL
                         foreground_mask=final_mask_pil, # Final (feathered) L mask PIL
                         prompt=effective_prompt,
                         **diff_params
                     )
                     if bg_pil is None: raise RuntimeError("Diffusion generation failed.")
                     background_rgb_np = np.array(bg_pil.convert('RGB'))
                 else:
                     raise ValueError(f"Unknown background type in dict: {bg_type}")
            else:
                 raise TypeError(f"Invalid background_spec type: {type(background_spec)}")

            if background_rgb_np is None:
                 raise ValueError("Background preparation resulted in None.")

            print("   Background prepared.")
        except Exception as e:
            print(f"Error preparing background: {e}")
            return False

        # 6. Add Shadow (Optional)
        # Start with the foreground RGBA numpy array (potentially feathered mask)
        foreground_with_shadow_np = foreground_rgba_np
        if add_shadow_flag:
            print("6. Adding drop shadow...")
            shadow_params = {
                'offset': (
                    self.config.get('shadow_offset_x', 5),
                    self.config.get('shadow_offset_y', 5)
                ),
                'blur_sigma': self.config.get('shadow_blur_sigma', 5.0),
                'opacity': self.config.get('shadow_opacity', 0.5),
                'shadow_color': self.config.get('shadow_color', (0, 0, 0))
            }
            print(f"   Shadow params: {shadow_params}")
            try:
                # Use the helper function, it returns a *new* RGBA array
                # with the shadow composited behind the original foreground
                foreground_with_shadow_np = add_soft_drop_shadow(foreground_rgba_np, **shadow_params)
                print("   Shadow added.")
            except Exception as e:
                print(f"Warning: Failed to add shadow: {e}. Proceeding without shadow.")
                # Keep the original foreground_rgba_np if shadow fails
                foreground_with_shadow_np = foreground_rgba_np
        else:
            print("6. Skipping drop shadow.")

        # 7. Combine Final Foreground (with shadow) and Background
        print("7. Combining final layers...")
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
            print("   Layers combined.")
        except Exception as e:
            print(f"Error combining layers: {e}")
            return False

        # 8. Save Result
        print(f"8. Saving final image to: {output_path}")
        # Convert to RGB before saving unless transparency is desired (e.g., PNG)
        save_mode = 'RGBA' if output_path.suffix.lower() == '.png' else 'RGB'
        success = save_image(final_image_pil.convert(save_mode), output_path)

        end_time = time.time()
        print(f"--- Processing finished in {end_time - start_time:.2f} seconds. Success: {success} ---")
        return success

# --- Example Usage (for testing within the module) ---
# if __name__ == '__main__':
#     print("Testing pipeline...")
#     # Setup paths relative to project root or use absolute paths
#     # Assumes running from project root where `data` and `results` exist
#     test_image = Path("../data/images/your_test_image.jpg") # CHANGE THIS
#     output_dir = Path("../results/pipeline_test")
#     output_dir.mkdir(parents=True, exist_ok=True)
# 
#     if not test_image.exists():
#         print(f"Test image not found: {test_image}")
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
#             print(f"\n--- Running Test Case {i+1} ---")
#             # Skip diffusion if not available
#             if isinstance(case['bg'], dict) and case['bg']['type'] == 'diffusion' and not pipeline.diffusion_generator:
#                 print("Skipping diffusion test case as DiffusionGenerator is not available.")
#                 continue
#                 
#             pipeline.process_image(
#                 image_path=test_image,
#                 output_path=case['output'],
#                 background_spec=case['bg'],
#                 prompt=case['prompt']
#             ) 