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

    Uses the 'rembg' library for initial segmentation and OpenCV for mask refinement.
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
            print(f"Segmenter initialized with model: {self.model_name}")
        except Exception as e:
            print(f"Error initializing rembg session for model {self.model_name}: {e}")
            # Optionally re-raise or handle depending on desired robustness
            raise ImportError("Failed to initialize rembg. Ensure it's installed and models are available.") from e

    def segment(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image],
        return_rgba: bool = False,
        refine: bool = True, # Add option to skip refinement
        refinement_config: Optional[Dict[str, Any]] = None, # Pass config directly
        save_intermediate: bool = False, # Option to save masks
        intermediate_dir: Optional[Union[str, Path]] = None, # Directory for saving
        output_basename: str = "segmented_item" # Basename for saved masks
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Segments the foreground object from an image.

        Args:
            image_input (Union[str, Path, np.ndarray, Image.Image]):
                Path to the image file, a NumPy array (H, W, C), or a PIL Image object.
            return_rgba (bool): If True, returns the RGBA image (foreground only)
                                along with the alpha mask. Otherwise, returns only the alpha mask.
            refine (bool): If True, applies mask refinement operations. Defaults to True.
            refinement_config (Optional[Dict[str, Any]]): Dictionary with refinement parameters
                overriding defaults from config.py (e.g., kernel sizes, iterations).
            save_intermediate (bool): If True, saves the raw and refined masks.
            intermediate_dir (Optional[Union[str, Path]]): Directory to save intermediate masks.
                                                        Required if save_intermediate is True.
            output_basename (str): Base filename for saving intermediate masks.

        Returns:
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
                - If return_rgba is False: A NumPy array representing the final alpha mask
                  (0-255, uint8), potentially refined.
                - If return_rgba is True: A tuple containing:
                    - The RGBA NumPy array (H, W, 4) of the segmented foreground.
                    - The final alpha mask NumPy array (H, W), potentially refined.

        Raises:
            TypeError: If the input type is not supported.
            FileNotFoundError: If the input is a path and the file does not exist.
            ValueError: If save_intermediate is True but intermediate_dir is not provided.
            Exception: For errors during the segmentation process.
        """
        if save_intermediate and not intermediate_dir:
            raise ValueError("intermediate_dir must be provided if save_intermediate is True.")

        if intermediate_dir:
            intermediate_dir = Path(intermediate_dir)
            intermediate_dir.mkdir(parents=True, exist_ok=True) # Ensure dir exists

        try:
            if isinstance(image_input, (str, Path)):
                image_path = Path(image_input)
                if not image_path.is_file():
                    raise FileNotFoundError(f"Image file not found: {image_path}")
                input_image = Image.open(image_path).convert('RGB')
            elif isinstance(image_input, np.ndarray):
                # Ensure input is in RGB order if it has 3 channels
                if image_input.ndim == 3 and image_input.shape[2] == 3:
                     # Assume BGR from OpenCV, convert to RGB for PIL
                     input_image = Image.fromarray(cv2.cvtColor(image_input.astype(np.uint8), cv2.COLOR_BGR2RGB))
                else: # Grayscale or other formats handled by PIL directly
                     input_image = Image.fromarray(image_input.astype(np.uint8))
                input_image = input_image.convert('RGB') # Ensure it's RGB for rembg
            elif isinstance(image_input, Image.Image):
                input_image = image_input.convert('RGB')
            else:
                raise TypeError(f"Unsupported input type: {type(image_input)}. "
                                "Expected str, Path, np.ndarray, or PIL.Image.")

            # Use the pre-initialized session
            output_image_pil = rembg.remove(input_image, session=self.session)

            # Extract raw alpha mask
            raw_alpha_mask = np.array(output_image_pil.getchannel(3)) # H, W, uint8

            # Save raw mask if requested
            if save_intermediate and intermediate_dir:
                save_image(
                    Image.fromarray(raw_alpha_mask),
                    intermediate_dir / f"{output_basename}_mask_raw.png"
                )

            final_mask = raw_alpha_mask
            if refine:
                # Get refinement parameters, merging defaults with overrides
                cfg = DEFAULT_CONFIG.copy()
                if refinement_config:
                    cfg.update(refinement_config)

                final_mask = self.refine_mask_from_config(raw_alpha_mask, cfg)

                # Save refined mask if requested
                if save_intermediate and intermediate_dir:
                    save_image(
                        Image.fromarray(final_mask),
                        intermediate_dir / f"{output_basename}_mask_refined.png"
                    )

            if return_rgba:
                # Apply the *final* mask (refined or raw) back to the original image
                # Create RGBA image from original RGB and final mask
                rgb_image_np = np.array(input_image)
                rgba_output = cv2.cvtColor(rgb_image_np, cv2.COLOR_RGB2RGBA)
                rgba_output[:, :, 3] = final_mask # Set alpha channel
                return rgba_output, final_mask
            else:
                return final_mask

        except FileNotFoundError as e:
            print(f"Error: {e}")
            raise
        except ValueError as e:
            print(f"Configuration Error: {e}")
            raise
        except Exception as e:
            print(f"Error during segmentation: {e}")
            # Consider more specific exception handling based on rembg errors
            raise RuntimeError(f"Segmentation failed for input. Error: {e}") from e

    @staticmethod
    def refine_mask_from_config(mask: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """
        Refines a mask using morphological operations defined in the config dictionary.

        Args:
            mask (np.ndarray): Grayscale alpha mask (0-255) or binary mask.
            config (Dict[str, Any]): Configuration dictionary containing keys like
                                     'mask_opening_kernel_size', 'mask_opening_iterations',
                                     'mask_dilation_kernel_size', 'mask_dilation_iterations'.

        Returns:
            np.ndarray: The refined mask (0-255, uint8).
        """
        if not isinstance(mask, np.ndarray):
             raise TypeError(f"Input mask must be a NumPy array, got {type(mask)}")

        refined_mask = mask.astype(np.uint8)
        if refined_mask.ndim == 3 and refined_mask.shape[2] == 1:
            refined_mask = refined_mask.squeeze()
        elif refined_mask.ndim != 2:
             raise ValueError(f"Input mask must be a single channel (grayscale or binary), shape was {mask.shape}")

        # --- Morphological Opening (Remove noise) ---
        open_k_size = config.get('mask_opening_kernel_size', 0)
        open_iter = config.get('mask_opening_iterations', 1)
        if open_k_size > 0 and open_k_size % 2 != 0: # Kernel size must be positive odd
            kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (open_k_size, open_k_size))
            refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel_open, iterations=open_iter)
        elif open_k_size > 0:
             print(f"Warning: mask_opening_kernel_size ({open_k_size}) must be odd, skipping opening.")

        # --- Morphological Dilation (Expand mask slightly) ---
        dilate_k_size = config.get('mask_dilation_kernel_size', 0)
        dilate_iter = config.get('mask_dilation_iterations', 1)
        if dilate_k_size > 0 and dilate_k_size % 2 != 0: # Kernel size must be positive odd
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_k_size, dilate_k_size))
            refined_mask = cv2.dilate(refined_mask, kernel_dilate, iterations=dilate_iter)
        elif dilate_k_size > 0:
             print(f"Warning: mask_dilation_kernel_size ({dilate_k_size}) must be odd, skipping dilation.")

        return refined_mask

    @staticmethod
    def refine_mask(
        mask: np.ndarray,
        operations: Optional[List[Dict[str, Any]]] = None
    ) -> np.ndarray:
        """
        DEPRECATED in favor of refine_mask_from_config for pipeline use.
        Applies morphological operations to refine a binary or alpha mask based on a list of operations.
        (Existing implementation remains)
        """
        if not isinstance(mask, np.ndarray):
             raise TypeError(f"Input mask must be a NumPy array, got {type(mask)}")

        if operations is None:
            operations = [
                {'type': 'close', 'kernel_size': 5, 'iterations': 1},
                {'type': 'dilate', 'kernel_size': 3, 'iterations': 1}
            ]

        # Ensure mask is 8-bit single channel for OpenCV operations
        if mask.ndim == 3:
             if mask.shape[2] == 1:
                 refined_mask = mask.squeeze().astype(np.uint8)
             else:
                 raise ValueError("Input mask must be a single channel (grayscale or binary).")
        elif mask.ndim == 2:
             refined_mask = mask.astype(np.uint8) # Convert boolean or other types
        else:
            raise ValueError(f"Input mask has unexpected dimensions: {mask.shape}")

        valid_ops = {'dilate': cv2.dilate, 'erode': cv2.erode, 'open': cv2.MORPH_OPEN, 'close': cv2.MORPH_CLOSE}

        for i, op in enumerate(operations):
            op_type = op.get('type')
            k_size = op.get('kernel_size')
            iterations = op.get('iterations', 1) # Default to 1 iteration

            if op_type not in valid_ops:
                raise ValueError(f"Invalid operation type '{op_type}' in operation {i}. "
                                 f"Valid types: {list(valid_ops.keys())}")
            if not isinstance(k_size, int) or k_size <= 0 or k_size % 2 == 0:
                raise ValueError(f"Invalid kernel_size '{k_size}' in operation {i}. "
                                 "Must be a positive odd integer.")
            if not isinstance(iterations, int) or iterations <= 0:
                raise ValueError(f"Invalid iterations '{iterations}' in operation {i}. "
                                 "Must be a positive integer.")

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))

            if op_type in ['open', 'close']:
                refined_mask = cv2.morphologyEx(refined_mask, valid_ops[op_type], kernel, iterations=iterations)
            else: # 'dilate', 'erode'
                refined_mask = valid_ops[op_type](refined_mask, kernel, iterations=iterations)

        return refined_mask

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