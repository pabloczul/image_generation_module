"""
Segmentation utilities using rembg and OpenCV for product background generation.
"""

import cv2
import numpy as np
from PIL import Image
import rembg
from pathlib import Path
from typing import Union, Optional, List, Dict, Any, Tuple
import os # Added for path joining
import logging # Added

# Use absolute import from src
from src.utils.data_io import save_image
from src.config import DEFAULT_CONFIG # Import default config for defaults

# Define supported rembg models
SUPPORTED_MODELS: Dict[str, str] = {
    'u2net': 'u2net',           # General purpose model
    'u2netp': 'u2netp',         # Lightweight version of u2net
    'silueta': 'silueta',       # Alternative model, sometimes better for certain objects
    'isnet-general-use': 'isnet-general-use' # Added IS-Net model
    # Add other models supported by rembg as needed
}

DEFAULT_MODEL = 'u2net'

class Segmenter:
    """
    Provides methods for segmenting product images to isolate foreground objects.

    Uses the 'rembg' library for initial segmentation. Refinement is handled externally.
    """
    def __init__(self, model_name: str = DEFAULT_MODEL):
        """
        Initializes the Segmenter.

        Args:
            model_name (str): The name of the rembg model to use.
                              Must be one of SUPPORTED_MODELS. Defaults to 'u2net'.
        
        Raises:
            ValueError: If the provided model_name is not supported.
            ImportError: If the 'rembg' library is not installed.
        """
        if model_name not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {model_name}. Supported models are: {list(SUPPORTED_MODELS.keys())}")
        
        self.model_name = model_name
        try:
            # Pre-initialize the session for potential reuse
            self.session = rembg.new_session(model_name=self.model_name)
            logging.info(f"Segmenter initialized with rembg model: {self.model_name}")
        except Exception as e:
            logging.exception(f"Error initializing rembg session for model {self.model_name}: {e}")
            # Optionally re-raise or handle depending on desired robustness
            raise ImportError("Failed to initialize rembg. Ensure it's installed and models are available.") from e

    def segment(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image],
        save_intermediate: bool = False, # Option to save masks
        intermediate_dir: Optional[Union[str, Path]] = None, # Directory for saving
        output_basename: str = "segmented_item" # Basename for saved masks
    ) -> Optional[np.ndarray]: # Changed return type
        """
        Segments the foreground object from an image using the configured rembg model.
        Returns only the raw alpha mask.

        Args:
            image_input (Union[str, Path, np.ndarray, Image.Image]):
                Path to the image file, a NumPy array (H, W, C), or a PIL Image object.
            save_intermediate (bool): If True, saves the raw mask.
            intermediate_dir (Optional[Union[str, Path]]): Directory to save intermediate mask.
                                                        Required if save_intermediate is True.
            output_basename (str): Base filename for saving intermediate mask.

        Returns:
            Optional[np.ndarray]: NumPy array representing the raw alpha mask (0-255, uint8),
                                  or None if segmentation fails.

        Raises:
            TypeError: If the input type is not supported.
            FileNotFoundError: If the input is a path and the file does not exist.
            ValueError: If save_intermediate is True but intermediate_dir is not provided.
            Exception: For errors during the segmentation process.
        """
        if save_intermediate and not intermediate_dir:
            logging.error("intermediate_dir must be provided if save_intermediate is True for Segmenter.")
            # Raise or return None depending on desired strictness
            # raise ValueError("intermediate_dir must be provided if save_intermediate is True.")
            return None 

        if intermediate_dir:
            intermediate_dir = Path(intermediate_dir)
            intermediate_dir.mkdir(parents=True, exist_ok=True) # Ensure dir exists

        try:
            # --- Input Image Handling --- (Simplified slightly)
            if isinstance(image_input, (str, Path)):
                image_path = Path(image_input)
                if not image_path.is_file():
                    logging.error(f"Image file not found: {image_path}")
                    raise FileNotFoundError(f"Image file not found: {image_path}")
                input_image = Image.open(image_path)
            elif isinstance(image_input, np.ndarray):
                 # Convert BGR to RGB if it looks like OpenCV image
                 if image_input.ndim == 3 and image_input.shape[2] == 3:
                     input_image = Image.fromarray(cv2.cvtColor(image_input.astype(np.uint8), cv2.COLOR_BGR2RGB))
                 else:
                     input_image = Image.fromarray(image_input.astype(np.uint8))
            elif isinstance(image_input, Image.Image):
                 input_image = image_input # Already a PIL Image
            else:
                logging.error(f"Unsupported input type: {type(image_input)}")
                raise TypeError(f"Unsupported input type: {type(image_input)}. Expected str, Path, np.ndarray, or PIL.Image.")
            
            # Ensure input is RGB for rembg consistency
            input_image_rgb = input_image.convert('RGB') 
            # --------------------------
            
            # Use the pre-initialized session
            logging.debug(f"Running rembg.remove with model {self.model_name}...")
            output_image_pil = rembg.remove(input_image_rgb, session=self.session)
            logging.debug("rembg.remove finished.")

            # Extract raw alpha mask
            raw_alpha_mask = np.array(output_image_pil.getchannel(3)) # H, W, uint8

            # Save raw mask if requested
            if save_intermediate and intermediate_dir:
                try:
                    save_path = intermediate_dir / f"{output_basename}_mask_raw.png"
                    save_image(Image.fromarray(raw_alpha_mask), save_path)
                    logging.info(f"Saved raw mask to {save_path}")
                except Exception as save_e:
                    logging.error(f"Failed to save raw mask to {save_path}: {save_e}")

            # Return only the raw mask
            return raw_alpha_mask

        except FileNotFoundError as e:
            # Already logged
            raise
        except ValueError as e:
            logging.error(f"Configuration or value error in segmentation: {e}", exc_info=True)
            raise
        except Exception as e:
            logging.exception(f"Error during segmentation process: {e}")
            # Consider more specific exception handling based on rembg errors
            # Return None or raise a custom exception?
            # For now, re-raise a RuntimeError to indicate failure
            raise RuntimeError(f"Segmentation failed. Error: {e}") from e

    # Removed refine_mask_from_config method (moved to filtering.py)
    
    # Deprecated refine_mask method can be removed or kept with clear deprecation warning
    @staticmethod
    def refine_mask(*args, **kwargs):
        raise DeprecationWarning("Segmenter.refine_mask is deprecated. Use refine_morphological from src.image.filtering instead.")

    # Evaluate mask quality method can remain if useful independently
    @staticmethod
    def evaluate_mask_quality(mask: np.ndarray) -> Dict[str, Any]:
        """
        Calculates quality metrics for a segmentation mask.

        Args:
            mask (np.ndarray): Grayscale alpha mask (0-255) or binary mask.

        Returns:
            Dict[str, Any]: A dictionary containing quality metrics:
                'coverage': Percentage of image area covered by the mask (float).
                'complexity': Ratio of perimeter to area (float). High values might 
                              indicate noise or fragmented masks.
                'contours': Number of distinct objects found (int).
                'is_valid': Heuristic check for basic validity (bool). True if coverage 
                            is between 5% and 95% and contours were found.
        """
        if not isinstance(mask, np.ndarray) or mask.ndim != 2:
            raise TypeError("Input mask must be a 2D NumPy array.")

        # Ensure mask is binary (0 or 255) for contour finding and calculations
        # Use a threshold (e.g., 127) if the input is grayscale alpha
        if mask.dtype != np.uint8:
             binary_mask = (mask > 127).astype(np.uint8) * 255 if np.issubdtype(mask.dtype, np.number) else mask.astype(np.uint8) * 255
        else:
             _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)


        if binary_mask.size == 0:
             return {'coverage': 0, 'complexity': 0, 'contours': 0, 'is_valid': False}
             
        coverage = np.count_nonzero(binary_mask) / binary_mask.size * 100

        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        num_contours = len(contours)
        if num_contours == 0:
            return {'coverage': 0, 'complexity': 0, 'contours': 0, 'is_valid': False}

        perimeter = sum(cv2.arcLength(cnt, True) for cnt in contours)
        area = np.count_nonzero(binary_mask) # Use count_nonzero for binary mask area

        complexity = perimeter / area if area > 0 else 0 # Avoid division by zero
        
        # Basic validity check: meaningful area coverage
        is_valid = 5.0 < coverage < 95.0 and area > 0 

        return {
            'coverage': round(coverage, 2),
            'complexity': round(complexity, 4),
            'contours': num_contours,
            'is_valid': is_valid
        }

# == Example Usage (keep commented out or remove for production) ==
# if __name__ == '__main__':
#     # Example usage requires an image file (e.g., 'product.jpg') 
#     # in the same directory or a specified path.
#     # Ensure 'rembg', 'opencv-python', 'numpy', 'Pillow' are installed.
#     
#     image_path = Path("../data/images/example_product.jpg") # Adjust path as needed
#     output_dir = Path("./segmentation_output")
#     output_dir.mkdir(exist_ok=True)
# 
#     if not image_path.exists():
#         print(f"Error: Example image not found at {image_path}")
#         print("Please provide a valid image path for testing.")
#     else:
#         try:
#             # Initialize with default model
#             segmenter = Segmenter() 
# 
#             # --- Test segmentation ---
#             print(f"Segmenting {image_path}...")
#             # Get RGBA and mask
#             rgba_output, alpha_mask = segmenter.segment(image_path, return_rgba=True) 
#             
#             # Save the alpha mask
#             mask_img = Image.fromarray(alpha_mask)
#             mask_save_path = output_dir / f"{image_path.stem}_mask.png"
#             mask_img.save(mask_save_path)
#             print(f"Saved alpha mask to {mask_save_path}")
#             
#             # Save the RGBA output (foreground only)
#             rgba_img = Image.fromarray(rgba_output)
#             rgba_save_path = output_dir / f"{image_path.stem}_rgba.png"
#             rgba_img.save(rgba_save_path)
#             print(f"Saved RGBA foreground to {rgba_save_path}")
# 
#             # --- Test mask refinement ---
#             print("Refining mask...")
#             # Custom operations example: more aggressive closing
#             # custom_ops = [{'type': 'close', 'kernel_size': 7, 'iterations': 2}]
#             # refined_mask = segmenter.refine_mask(alpha_mask, operations=custom_ops)
#             refined_mask = segmenter.refine_mask(alpha_mask) # Use default ops
#             
#             refined_mask_img = Image.fromarray(refined_mask)
#             refined_save_path = output_dir / f"{image_path.stem}_mask_refined.png"
#             refined_mask_img.save(refined_save_path)
#             print(f"Saved refined mask to {refined_save_path}")
# 
#             # --- Test mask evaluation ---
#             print("Evaluating original mask...")
#             original_metrics = segmenter.evaluate_mask_quality(alpha_mask)
#             print(f"Original mask metrics: {original_metrics}")
#             
#             print("Evaluating refined mask...")
#             refined_metrics = segmenter.evaluate_mask_quality(refined_mask)
#             print(f"Refined mask metrics: {refined_metrics}")
# 
#         except (ImportError, FileNotFoundError, ValueError, RuntimeError) as e:
#             print(f"An error occurred: {e}")
#         except Exception as e:
#              print(f"An unexpected error occurred: {e}")

# Clean up old functions if they existed at the top level
# (No longer needed as they are now methods of the Segmenter class)
# def segment_product(...): ...
# def refine_mask(...): ...
# def evaluate_mask_quality(...): ... 