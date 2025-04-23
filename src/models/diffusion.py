"""
Diffusion-based background generation using Stable Diffusion with ControlNet Inpainting.

Relies on the Hugging Face `diffusers` library.
Ensure necessary dependencies are installed: 
`pip install torch diffusers transformers accelerate opencv-python`
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, Optional, Dict, Any, Tuple
import warnings
import cv2 # Imported for Canny edge detection
import logging # Added

try:
    from diffusers import (
        StableDiffusionControlNetInpaintPipeline, 
        ControlNetModel, 
        DDIMScheduler,
        UniPCMultistepScheduler # Added alternative scheduler
    )
    from diffusers.utils import load_image
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    warnings.warn("Diffusers library not found. DiffusionGenerator will not be functional. "
                  "Please install with: pip install diffusers torch transformers accelerate")
    # Define dummy classes or raise errors later if not available
    StableDiffusionControlNetInpaintPipeline = None
    ControlNetModel = None
    DDIMScheduler = None
    UniPCMultistepScheduler = None
    load_image = None

# --- Constants and Configuration ---
DEFAULT_SD_MODEL = "runwayml/stable-diffusion-v1-5"
DEFAULT_INPAINT_MODEL = "runwayml/stable-diffusion-inpainting" # Alternative base model
DEFAULT_CONTROLNET_SEG_MODEL = "lllyasviel/sd-controlnet-seg"
DEFAULT_CONTROLNET_CANNY_MODEL = "lllyasviel/sd-controlnet-canny"
# Add other ControlNet models as needed (depth, openpose, etc.)

# Mapping for easier selection
CONTROLNET_MODELS = {
    'seg': DEFAULT_CONTROLNET_SEG_MODEL,
    'canny': DEFAULT_CONTROLNET_CANNY_MODEL,
    # 'depth': 'lllyasviel/sd-controlnet-depth',
}

SUPPORTED_CONDITION_TYPES = list(CONTROLNET_MODELS.keys())

DEFAULT_SCHEDULER = "DDIM" # Options: "DDIM", "UniPC"
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Diffusion Generator Class ---

class DiffusionGenerator:
    """
    Manages Stable Diffusion ControlNet Inpainting pipelines for background generation.

    Loads models upon initialization for efficiency.
    """
    def __init__(
        self,
        sd_model_id: str = DEFAULT_SD_MODEL,
        # Consider using DEFAULT_INPAINT_MODEL as base sometimes
        controlnet_type: str = 'seg', # Default conditioning
        custom_controlnet_id: Optional[str] = None, # Override default CN model
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        scheduler_type: str = DEFAULT_SCHEDULER,
        enable_cpu_offload: bool = True # Good for memory, might slow down
    ):
        """
        Initializes the DiffusionGenerator, loading models and setting up the pipeline.
    
    Args:
            sd_model_id (str): Hugging Face model ID for the base Stable Diffusion model.
            controlnet_type (str): The type of ControlNet to use ('seg', 'canny', etc.).
                                   Must be a key in `CONTROLNET_MODELS`.
            custom_controlnet_id (Optional[str]): Provide a specific ControlNet model ID
                                                to override the default for the type.
            device (Optional[str]): Device to run inference on ("cuda", "cpu"). 
                                    Defaults to GPU if available, otherwise CPU.
            torch_dtype (Optional[torch.dtype]): Data type for model weights 
                                               (e.g., torch.float16 for faster inference
                                               on compatible GPUs). Defaults based on device.
            scheduler_type (str): Scheduler type ("DDIM", "UniPC").
            enable_cpu_offload (bool): If True and on CUDA, enables model offloading to CPU
                                       to save VRAM.

        Raises:
            ImportError: If the `diffusers` library is not installed.
            ValueError: If an invalid `controlnet_type` or `scheduler_type` is provided.
            RuntimeError: If model loading fails.
        """
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("DiffusionGenerator requires the 'diffusers' library. Please install it.")

        if controlnet_type not in CONTROLNET_MODELS:
            raise ValueError(f"Unsupported controlnet_type: '{controlnet_type}'. "
                             f"Supported types: {list(CONTROLNET_MODELS.keys())}")

        self.device = device if device else DEFAULT_DEVICE
        self.controlnet_type = controlnet_type
        
        # Determine ControlNet model ID
        self.controlnet_id = custom_controlnet_id if custom_controlnet_id else CONTROLNET_MODELS[controlnet_type]

        # Determine dtype based on device
        if torch_dtype is None:
            self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        else:
            self.torch_dtype = torch_dtype

        # Use logging instead of print
        logging.info(f"Initializing DiffusionGenerator on device: {self.device} with dtype: {self.torch_dtype}")
        logging.info(f"Loading Stable Diffusion model: {sd_model_id}")
        logging.info(f"Loading ControlNet ({controlnet_type}): {self.controlnet_id}")

        try:
            self.controlnet = ControlNetModel.from_pretrained(
                self.controlnet_id, 
                torch_dtype=self.torch_dtype
            )

            self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                sd_model_id,
                controlnet=self.controlnet,
                torch_dtype=self.torch_dtype,
                safety_checker=None # Disable safety checker
            )

            # Setup scheduler
            if scheduler_type.upper() == "DDIM":
                 self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
            elif scheduler_type.upper() == "UNIPC":
                 self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
            else:
                 raise ValueError(f"Unsupported scheduler_type: {scheduler_type}. Choose 'DDIM' or 'UniPC'.")

            self.pipe = self.pipe.to(self.device)

            if self.device == "cuda" and enable_cpu_offload:
                 logging.info("Enabling CPU offload for model components.") # Use logging
                 self.pipe.enable_model_cpu_offload()
            elif self.device == "cuda":
                 # Potential memory optimization if offload is not used
                 # self.pipe.enable_vae_slicing() # Consider if needed
                 pass

            logging.info("Diffusion pipeline initialized successfully.") # Use logging

        except Exception as e:
            logging.error(f"Error loading diffusion models or setting up pipeline: {e}", exc_info=True) # Use logging, add traceback
            raise RuntimeError(f"Failed to initialize DiffusionGenerator. Error: {e}") from e
            
    @staticmethod
    def _prepare_image_mask_pair(
        image: Union[Image.Image, np.ndarray, str, Path],
        mask: Union[Image.Image, np.ndarray, str, Path]
    ) -> Tuple[Optional[Image.Image], Optional[Image.Image]]:
        """Loads and prepares image and foreground mask (converts to PIL)."""
        
        def load_pil(img_input, mode='RGB'):
            if isinstance(img_input, (str, Path)):
                if not Path(img_input).exists(): raise FileNotFoundError(f"{img_input}")
                return Image.open(img_input).convert(mode)
            elif isinstance(img_input, np.ndarray):
                # Basic check for channel order might be needed if both RGB/BGR numpy arrays expected
                if img_input.ndim == 3 and img_input.shape[2] == 3 and mode=='RGB':
                    return Image.fromarray(img_input.astype(np.uint8))
                elif img_input.ndim == 2 and mode=='L': # Grayscale mask
                     return Image.fromarray(img_input.astype(np.uint8))
                else: # Attempt conversion if possible
                    try: return Image.fromarray(img_input.astype(np.uint8)).convert(mode)
                    except Exception as e: raise TypeError(f"Cannot convert numpy array to PIL ({mode}): {e}")
            elif isinstance(img_input, Image.Image):
                return img_input.convert(mode)
            else:
                raise TypeError(f"Unsupported input type for image/mask: {type(img_input)}")
                
        try:
            image_pil = load_pil(image, 'RGB')
            mask_pil = load_pil(mask, 'L') # Load mask as grayscale
            
            if image_pil.size != mask_pil.size:
                raise ValueError(f"Image ({image_pil.size}) and mask ({mask_pil.size}) dimensions must match.")
                
            return image_pil, mask_pil
        except Exception as e:
            logging.error(f"Error preparing image/mask: {e}", exc_info=True) # Use logging
            return None, None
            
    @staticmethod
    def _prepare_control_image(
        image: Image.Image, 
        foreground_mask: Image.Image, 
        condition_type: str
    ) -> Optional[Image.Image]:
        """
        Prepares the conditioning image for ControlNet based on the type.
    
        Args:
            image (Image.Image): Input RGB image.
            foreground_mask (Image.Image): Grayscale mask (non-zero for foreground).
            condition_type (str): Type of conditioning ('seg', 'canny').
        
        Returns:
            Optional[Image.Image]: The conditioning image, or None if fails.
        """
        try:
            # Ensure mask is binary (0/255) for consistent processing
            binary_mask_pil = foreground_mask.point(lambda p: 255 if p > 127 else 0)

            if condition_type == 'seg':
                # Black foreground (0), White background (255) -> Invert binary mask
                inverted_mask = Image.eval(binary_mask_pil, lambda p: 255 - p)
                # Create white canvas and paste black foreground using the inverted mask
                control_image = Image.new('RGB', image.size, (255, 255, 255))
                control_image.paste((0, 0, 0), mask=binary_mask_pil) # Paste black using original FG mask
                return control_image

            elif condition_type == 'canny':
                img_np = np.array(image.convert('L')) # Canny works on grayscale
                # Detect edges (adjust thresholds as needed)
                edges_np = cv2.Canny(img_np, 100, 200)
                control_image = Image.fromarray(edges_np).convert('RGB')
                return control_image
                
            # Placeholder for Depth - Requires a dedicated model
            # elif condition_type == 'depth':
            #    # Load depth model, predict depth map, convert to PIL Image
            #    # Example using distance transform as placeholder:
            #    mask_np = np.array(binary_mask_pil)
            #    dist = cv2.distanceTransform(mask_np, cv2.DIST_L2, 5) # Distance from foreground
            #    dist = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            #    control_image = Image.fromarray(dist).convert('RGB')
            #    return control_image
            
            else:
                warnings.warn(f"Condition type '{condition_type}' currently unsupported for control image generation.")
                return None
        except Exception as e:
            logging.error(f"Error preparing control image ({condition_type}): {e}", exc_info=True) # Use logging
            return None

    def generate(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image],
        foreground_mask: Union[str, Path, np.ndarray, Image.Image], # Mask where FG is non-zero
        prompt: str,
        negative_prompt: str = "low quality, bad quality, blurry, deformed, text, words, signature",
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: float = 0.75, # ControlNet strength
        generator: Optional[torch.Generator] = None # For reproducibility
    ) -> Optional[Image.Image]:
        """
        Generates a new background using the initialized ControlNet Inpainting pipeline.

        Args:
            image_input: The input image (path, array, or PIL).
            foreground_mask: The mask indicating the foreground object (non-zero pixels).
                             Background (zero pixels) will be inpainted.
            prompt (str): Text prompt describing the desired background or scene.
            negative_prompt (str): Text prompt for undesired elements.
            num_inference_steps (int): Number of diffusion steps.
            guidance_scale (float): Scale for guidance loss (higher means follows prompt more).
            controlnet_conditioning_scale (float): Weight of the ControlNet conditioning.
            generator (Optional[torch.Generator]): PyTorch generator for deterministic results.

        Returns:
            Optional[Image.Image]: The generated image with the new background, or None if fails.
        """
        image_pil, mask_pil = self._prepare_image_mask_pair(image_input, foreground_mask)
        if image_pil is None or mask_pil is None:
            # Error already logged by _prepare_image_mask_pair
            return None

        # Prepare ControlNet conditioning image based on the initialized type
        control_image = self._prepare_control_image(image_pil, mask_pil, self.controlnet_type)
        if control_image is None:
            logging.error(f"Failed to prepare control image for type '{self.controlnet_type}'. Aborting generation.") # Use logging
            return None

        # Prepare mask for inpainting: 0=Keep, 255=Inpaint
        # Invert the foreground mask
        inpaint_mask_pil = Image.eval(mask_pil.convert('L'), lambda p: 255 if p < 128 else 0)
        
        # Resize all inputs to be multiples of 8 (required by SD)
        width, height = image_pil.size
        new_width = (width // 8) * 8
        new_height = (height // 8) * 8
    
        if new_width == 0 or new_height == 0:
            logging.error(f"Error: Image dimensions ({width}x{height}) are too small, resulting in 0 after rounding to multiple of 8.") # Use logging
            return None
            
        logging.info(f"Resizing inputs from ({width}x{height}) to ({new_width}x{new_height}) for diffusion.") # Use logging
        image_pil = image_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
        inpaint_mask_pil = inpaint_mask_pil.resize((new_width, new_height), Image.Resampling.NEAREST)
        control_image = control_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        try:
            logging.info(f"Generating background with prompt: '{prompt}'") # Use logging
            # Run the pipeline
            result_image = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image_pil,
                mask_image=inpaint_mask_pil,
                control_image=control_image,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                generator=generator,
                # strength=1.0, # For inpainting, strength is often 1.0
                ).images[0]
    
            # Resize back to original size? Optional, depends on desired output.
            # result_image = result_image.resize((width, height), Image.Resampling.LANCZOS)

            logging.info("Background generation finished.") # Use logging
            return result_image

        except Exception as e:
            logging.error(f"Error during diffusion pipeline execution: {e}", exc_info=True) # Use logging
            # Potentially catch specific errors like OOM
            if "CUDA out of memory" in str(e):
                logging.error("CUDA Out of Memory error. Try reducing image size, using CPU offload, or a smaller model.") # Use logging
            return None

# == Example Usage ==
# if __name__ == '__main__':
#     if not DIFFUSERS_AVAILABLE:
#         print("Cannot run example: diffusers library not available.")
#     else:
#         image_path = Path("../data/images/example_product.jpg") # Adjust path
#         mask_path = Path("../data/masks/example_product_mask.png") # Adjust path
#         output_dir = Path("./diffusion_output")
#         output_dir.mkdir(exist_ok=True)
# 
#         if not image_path.exists() or not mask_path.exists():
#             print(f"Error: Example image ({image_path}) or mask ({mask_path}) not found.")
#         else:
#             try:
#                 # --- Initialize Generator (Choose ControlNet type) ---
#                 # generator_seg = DiffusionGenerator(controlnet_type='seg') # Uses seg model
#                 generator_canny = DiffusionGenerator(controlnet_type='canny') # Uses canny model
#                 active_generator = generator_canny # Choose which one to run
#                 
#                 # --- Define Prompt ---
#                 # prompt = "A modern minimalist apartment living room, clean, bright daylight"
#                 prompt = "A clean white photography studio background, soft shadows"
#                 
#                 # --- Set seed for reproducibility ---
#                 seed = 42 
#                 pytorch_generator = torch.Generator(device=active_generator.device).manual_seed(seed)
#                 
#                 # --- Generate Image ---
#                 print(f"\nStarting generation with {active_generator.controlnet_type} ControlNet...")
#                 generated_image = active_generator.generate(
#                     image_input=image_path,
#                     foreground_mask=mask_path,
#                     prompt=prompt,
#                     negative_prompt="ugly, deformed, watermark, text, signature, blurry, low quality",
#                     num_inference_steps=30,
#                     guidance_scale=7.5,
#                     controlnet_conditioning_scale=0.8, # Adjust strength of controlnet
#                     generator=pytorch_generator
#                 )
# 
#                 # --- Save Output ---
#                 if generated_image:
#                     save_filename = f"{image_path.stem}_generated_bg_{active_generator.controlnet_type}_seed{seed}.png"
#                     save_path = output_dir / save_filename
#                     generated_image.save(save_path)
#                     print(f"Saved generated image to: {save_path}")
#                 else:
#                     print("Image generation failed.")
# 
#             except (ImportError, FileNotFoundError, ValueError, RuntimeError) as e:
#                 print(f"An error occurred: {e}")
#             except Exception as e:
#                 print(f"An unexpected error occurred: {e}")

# Clean up old functions
# def prepare_condition_image(...): ...
# def generate_background_diffusion(...): ... 