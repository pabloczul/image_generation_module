"""
Image quality assessment utilities for the product background generation pipeline.

Provides tools to check image resolution, blur, contrast, and background complexity,
and recommend processing strategies based on these metrics.
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, Tuple, Dict, Any, Optional, List
import warnings

# --- Constants and Configuration ---
DEFAULT_MIN_RESOLUTION: Tuple[int, int] = (300, 300)
DEFAULT_BLUR_THRESHOLD: float = 100.0  # Laplacian variance
DEFAULT_CONTRAST_THRESHOLD: float = 30.0 # Standard deviation of grayscale image
DEFAULT_BG_COMPLEXITY_THRESHOLD: float = 0.1 # Edge density (edges per pixel)

# Define quality levels
QUALITY_EXCELLENT = 'excellent'
QUALITY_GOOD = 'good'
QUALITY_FAIR = 'fair'
QUALITY_POOR = 'poor'

# --- Helper Functions ---

def _load_image(image_input: Union[str, Path, np.ndarray, Image.Image]) -> Tuple[Optional[Image.Image], Optional[np.ndarray]]:
    """Loads an image into PIL and OpenCV formats."""
    pil_image: Optional[Image.Image] = None
    cv_image: Optional[np.ndarray] = None

    try:
        if isinstance(image_input, (str, Path)):
            image_path = Path(image_input)
            if not image_path.is_file():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            pil_image = Image.open(image_path).convert('RGB')
            # Convert PIL (RGB) to OpenCV (BGR)
            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 3 and image_input.shape[2] == 3:
                # Assume input is BGR if it's a typical OpenCV image, else assume RGB
                # This distinction might be ambiguous without more context.
                # For simplicity, let's assume BGR is standard for numpy input here.
                cv_image = image_input.copy()
            if cv_image.dtype != np.uint8:
                # Attempt conversion, warn if precision loss
                if np.max(cv_image) > 255 or np.min(cv_image) < 0:
                    warnings.warn("NumPy array values outside [0, 255] range, potential precision loss during uint8 conversion.")
                cv_image = np.clip(cv_image, 0, 255).astype(np.uint8)
                pil_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            elif len(image_input.shape) == 2: # Grayscale
                cv_image = cv2.cvtColor(image_input.astype(np.uint8), cv2.COLOR_GRAY2BGR)
                pil_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            else:
                 raise ValueError("Input NumPy array must be a 3-channel color image (H, W, 3) or grayscale (H,W).")
        elif isinstance(image_input, Image.Image):
            pil_image = image_input.convert('RGB')
            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        else:
            raise TypeError(f"Unsupported input type: {type(image_input)}. "
                            "Expected str, Path, np.ndarray, or PIL.Image.")
        return pil_image, cv_image
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None, None
    except Exception as e:
        print(f"Error loading or converting image: {e}")
        return None, None

# --- Image Assessor Class ---

class ImageAssessor:
    """
    Assesses image quality based on resolution, blur, contrast, and background complexity.
    """
    def __init__(
        self,
        min_resolution: Tuple[int, int] = DEFAULT_MIN_RESOLUTION,
        blur_threshold: float = DEFAULT_BLUR_THRESHOLD,
        contrast_threshold: float = DEFAULT_CONTRAST_THRESHOLD,
        bg_complexity_threshold: float = DEFAULT_BG_COMPLEXITY_THRESHOLD
    ):
        """
        Initializes the ImageAssessor with quality thresholds.

        Args:
            min_resolution (Tuple[int, int]): Minimum acceptable (width, height).
            blur_threshold (float): Minimum Laplacian variance. Lower means more blurry.
            contrast_threshold (float): Minimum standard deviation of grayscale pixel intensities.
            bg_complexity_threshold (float): Maximum edge density in background.
        """
        if not (isinstance(min_resolution, tuple) and len(min_resolution) == 2 and 
                all(isinstance(dim, int) and dim > 0 for dim in min_resolution)):
            raise ValueError("min_resolution must be a tuple of two positive integers (width, height).")
        if not isinstance(blur_threshold, (int, float)) or blur_threshold < 0:
             raise ValueError("blur_threshold must be a non-negative number.")
        if not isinstance(contrast_threshold, (int, float)) or contrast_threshold < 0:
             raise ValueError("contrast_threshold must be a non-negative number.")
        if not isinstance(bg_complexity_threshold, (int, float)) or bg_complexity_threshold < 0:
             raise ValueError("bg_complexity_threshold must be a non-negative number.")
             
        self.min_resolution = min_resolution
        self.blur_threshold = blur_threshold
        self.contrast_threshold = contrast_threshold
        self.bg_complexity_threshold = bg_complexity_threshold

    def _check_resolution(self, pil_image: Image.Image) -> Dict[str, Any]:
        """Checks if image resolution meets the minimum requirement."""
        width, height = pil_image.size
        passed = width >= self.min_resolution[0] and height >= self.min_resolution[1]
        return {'width': width, 'height': height, 'passed': passed}
    
    def _calculate_blur(self, cv_image: np.ndarray) -> Dict[str, Any]:
        """Calculates image blur using Laplacian variance."""
        try:
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            blur_value = cv2.Laplacian(gray, cv2.CV_64F).var()
            passed = blur_value >= self.blur_threshold
            return {'value': round(blur_value, 2), 'passed': passed}
        except cv2.error as e:
             print(f"OpenCV error during blur calculation: {e}")
             return {'value': 0.0, 'passed': False, 'error': str(e)}
        except Exception as e:
            print(f"Unexpected error during blur calculation: {e}")
            return {'value': 0.0, 'passed': False, 'error': str(e)}
    
    def _calculate_contrast(self, cv_image: np.ndarray) -> Dict[str, Any]:
        """Calculates image contrast using standard deviation of grayscale pixels."""
        try:
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            contrast_value = gray.std()
            passed = contrast_value >= self.contrast_threshold
            return {'value': round(contrast_value, 2), 'passed': passed}
        except cv2.error as e:
             print(f"OpenCV error during contrast calculation: {e}")
             return {'value': 0.0, 'passed': False, 'error': str(e)}
        except Exception as e:
            print(f"Unexpected error during contrast calculation: {e}")
            return {'value': 0.0, 'passed': False, 'error': str(e)}

    def assess_quality(self, image_input: Union[str, Path, np.ndarray, Image.Image]) -> Dict[str, Any]:
        """
        Performs a comprehensive quality assessment of the input image.

        Args:
            image_input: Path to image, Path object, NumPy array (BGR or Grayscale),
                         or PIL Image object.

        Returns:
            Dict[str, Any]: Assessment results including resolution, blur, contrast,
                            overall quality score, processability flag, and messages.
        """
        pil_image, cv_image = _load_image(image_input)

        if pil_image is None or cv_image is None:
            return {
                'resolution': {'passed': False, 'error': 'Image loading failed'},
                'blur': {'passed': False, 'error': 'Image loading failed'},
                'contrast': {'passed': False, 'error': 'Image loading failed'},
                'overall_quality': QUALITY_POOR,
                'is_processable': False,
                'message': "Failed to load or process the image."
            }

        results = {
            'resolution': self._check_resolution(pil_image),
            'blur': self._calculate_blur(cv_image),
            'contrast': self._calculate_contrast(cv_image),
            'overall_quality': QUALITY_POOR, # Default
            'is_processable': False,      # Default
            'message': ""
        }

        # Aggregate messages for failures
        messages = []
        if not results['resolution']['passed']:
            messages.append(f"Resolution ({results['resolution']['width']}x{results['resolution']['height']}) below minimum ({self.min_resolution[0]}x{self.min_resolution[1]}).")
        if not results['blur']['passed']:
            messages.append(f"Image may be too blurry (Laplacian Var: {results['blur']['value']:.2f}, Threshold: {self.blur_threshold:.2f}).")
        if not results['contrast']['passed']:
            messages.append(f"Image may have low contrast (Std Dev: {results['contrast']['value']:.2f}, Threshold: {self.contrast_threshold:.2f}).")
    
        # Determine overall quality and processability based on checks
        res_ok = results['resolution']['passed']
        blur_ok = results['blur']['passed']
        contrast_ok = results['contrast']['passed']

        if res_ok and blur_ok and contrast_ok:
            results['overall_quality'] = QUALITY_EXCELLENT
            results['is_processable'] = True
            messages.append("Image quality is excellent.")
        elif res_ok and (blur_ok or contrast_ok):
            results['overall_quality'] = QUALITY_GOOD
            results['is_processable'] = True
            messages.append("Image quality is good.")
        elif res_ok:
            results['overall_quality'] = QUALITY_FAIR
            results['is_processable'] = True # Still processable, but maybe poor results
            messages.append("Image quality is fair (low blur and contrast). Results may be suboptimal.")
        else: # Resolution failed
            results['overall_quality'] = QUALITY_POOR
            results['is_processable'] = False
            messages.append("Image quality is poor (resolution too low).")
    
        results['message'] = " ".join(messages).strip()
        return results

    def check_background_complexity(
        self, 
        image_input: Union[str, Path, np.ndarray, Image.Image],
        foreground_mask: np.ndarray
    ) -> Dict[str, Any]:
        """
        Checks the complexity of the image background using edge detection.
        Requires a foreground mask (where foreground is non-zero).
    
        Args:
            image_input: Path to image, Path object, NumPy array (BGR or Grayscale),
                         or PIL Image object.
            foreground_mask (np.ndarray): A binary or grayscale mask where the 
                                          foreground object is non-zero (e.g., 255) 
                                          and the background is zero.
        
        Returns:
            Dict[str, Any]: Background complexity metrics including edge count,
                            area, density, and a complexity flag.
        """
        _pil_image, cv_image = _load_image(image_input)

        if cv_image is None:
            return {'is_complex': False, 'error': 'Image loading failed', 'message': 'Could not assess background complexity.'}

        if not isinstance(foreground_mask, np.ndarray) or foreground_mask.ndim != 2:
             raise TypeError("foreground_mask must be a 2D NumPy array.")
             
        if foreground_mask.shape[:2] != cv_image.shape[:2]:
            raise ValueError("Mask dimensions must match image dimensions.")

        # Ensure mask is binary (0 for background, 255 for foreground)
        if np.max(foreground_mask) == 1: # Handle boolean masks
             binary_fg_mask = foreground_mask.astype(np.uint8) * 255
        else:
             _, binary_fg_mask = cv2.threshold(foreground_mask.astype(np.uint8), 1, 255, cv2.THRESH_BINARY)

        # Invert the mask to get the background
        bg_mask = cv2.bitwise_not(binary_fg_mask)
    
        try:
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    
            # Apply Canny edge detection on the entire image
            # Parameters might need tuning based on image characteristics
            edges = cv2.Canny(gray, 50, 150) 
    
            # Isolate edges only in the background region
            background_edges = cv2.bitwise_and(edges, edges, mask=bg_mask)
    
            edge_count = cv2.countNonZero(background_edges)
            bg_area = cv2.countNonZero(bg_mask)
    
            edge_density = (edge_count / bg_area) if bg_area > 0 else 0.0
            is_complex = edge_density > self.bg_complexity_threshold
            
            message = (
                 f"Background complexity assessed. Edge Density: {edge_density:.4f}. "
                 f"{'Complex background detected.' if is_complex else 'Background complexity acceptable.'}"
            )
    
            return {
                'edge_count': edge_count,
                'background_area': bg_area,
                        'edge_density': round(edge_density, 4),
                'is_complex': is_complex,
                        'message': message
                    }
        except cv2.error as e:
            print(f"OpenCV error during background complexity check: {e}")
            return {'is_complex': False, 'error': str(e), 'message': 'Failed background complexity check.'}
        except Exception as e:
             print(f"Unexpected error during background complexity check: {e}")
             return {'is_complex': False, 'error': str(e), 'message': 'Failed background complexity check.'}

    def recommend_processing_strategy(
        self, 
        quality_assessment: Dict[str, Any],
        bg_complexity: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Recommends a processing strategy based on image quality and background complexity.
    
        Args:
            quality_assessment (Dict[str, Any]): Results from assess_quality().
            bg_complexity (Optional[Dict[str, Any]]): Results from check_background_complexity().
        
        Returns:
            Dict[str, Any]: Recommended strategy including processability, segmentation model,
                            mask refinement operations, suggested background type, and messages.
        """
        strategy = {
                'is_processable': quality_assessment.get('is_processable', False),
                'segmentation_model': 'u2net',  # Default
                'mask_refinement_ops': [
                    {'type': 'close', 'kernel_size': 5, 'iterations': 1},
                    {'type': 'dilate', 'kernel_size': 3, 'iterations': 1}
                ], # Default
                'suggested_background_type': 'gradient', # Default
                'messages': []
        }
    
        # Add assessment messages
        if quality_assessment.get('message'):
             strategy['messages'].append(quality_assessment['message']) 

        if not strategy['is_processable']:
            strategy['messages'].append("Cannot recommend strategy: Image deemed not processable.")
            return strategy
    
        # --- Rule-based Recommendations --- 
        quality = quality_assessment.get('overall_quality', QUALITY_POOR)
        
        # 1. Segmentation Model Choice (Example: Use better model for fair quality)
        if quality == QUALITY_FAIR:
            # Consider a potentially more robust model if available and needed
            # strategy['segmentation_model'] = 'isnet-general-use' 
            strategy['messages'].append("Consider reviewing segmentation as quality is fair.")
        # else: keep default 'u2net'

        # 2. Mask Refinement Strategy
        blur_passed = quality_assessment.get('blur', {}).get('passed', False)
        contrast_passed = quality_assessment.get('contrast', {}).get('passed', False)
        blur_value = quality_assessment.get('blur', {}).get('value', 0)
        
        if not blur_passed and blur_value < self.blur_threshold / 2: # Significantly blurry
            strategy['mask_refinement_ops'] = [
                {'type': 'close', 'kernel_size': 7, 'iterations': 1},
                {'type': 'dilate', 'kernel_size': 5, 'iterations': 1}
            ]
            strategy['messages'].append("Applying stronger mask refinement due to potential blur.")
        elif not contrast_passed:
            strategy['mask_refinement_ops'] = [
                {'type': 'close', 'kernel_size': 9, 'iterations': 1},
                {'type': 'erode', 'kernel_size': 3, 'iterations': 1},
                {'type': 'dilate', 'kernel_size': 5, 'iterations': 1}
            ]
            strategy['messages'].append("Applying stronger mask refinement due to low contrast.")
        # else: Keep default refinement

        # 3. Background Type Recommendation
        width = quality_assessment.get('resolution', {}).get('width', 0)
        height = quality_assessment.get('resolution', {}).get('height', 0)
        if width < 800 or height < 800:
             strategy['suggested_background_type'] = 'solid'
             strategy['messages'].append("Suggesting solid background for lower resolution image.")
        # else: keep default gradient

        # 4. Consider Background Complexity (if provided)
        if bg_complexity and bg_complexity.get('is_complex', False):
            strategy['messages'].append("Complex background detected. Segmentation might require careful review or alternative methods (e.g., diffusion inpainting if applicable).")
            # Optionally, could change segmentation model or suggest manual review
            # strategy['segmentation_model'] = 'some_other_model_for_complex_bg'
        if bg_complexity and bg_complexity.get('message'):
            strategy['messages'].append(bg_complexity['message']) # Add complexity check message
    
        return strategy 

# == Example Usage (keep commented out or remove for production) ==
# if __name__ == '__main__':
#     # Example usage requires an image file (e.g., 'product.jpg') 
#     # and potentially a mask file (e.g., 'product_mask.png')
#     # Ensure 'opencv-python', 'numpy', 'Pillow' are installed.
# 
#     image_path = Path("../data/images/example_product.jpg") # Adjust path
#     mask_path = Path("../data/masks/example_product_mask.png") # Adjust path (Create a dummy mask if needed)
#     output_dir = Path("./quality_output")
#     output_dir.mkdir(exist_ok=True)
# 
#     if not image_path.exists():
#         print(f"Error: Example image not found at {image_path}")
#     else:
#         try:
#             # Initialize assessor with default thresholds
#             assessor = ImageAssessor()
# 
#             # --- Test Quality Assessment ---
#             print(f"Assessing quality for {image_path}...")
#             quality_results = assessor.assess_quality(image_path)
#             print("\nQuality Assessment Results:")
#             import json
#             print(json.dumps(quality_results, indent=2))
# 
#             # --- Test Background Complexity (Requires a mask) ---
#             if mask_path.exists():
#                 print(f"\nChecking background complexity using mask {mask_path}...")
#                 try:
#                     foreground_mask = np.array(Image.open(mask_path).convert('L')) # Load as grayscale
#                     complexity_results = assessor.check_background_complexity(image_path, foreground_mask)
#                     print("\nBackground Complexity Results:")
#                     print(json.dumps(complexity_results, indent=2))
#                 except Exception as e:
#                      print(f"Could not load or process mask {mask_path}: {e}")
#                      complexity_results = None # Set to None if mask fails
#             else:
#                 print(f"\nMask file not found at {mask_path}. Skipping background complexity check.")
#                 complexity_results = None
# 
#             # --- Test Strategy Recommendation ---
#             print("\nRecommending processing strategy...")
#             strategy = assessor.recommend_processing_strategy(quality_results, complexity_results)
#             print("\nRecommended Strategy:")
#             print(json.dumps(strategy, indent=2))
# 
#         except (FileNotFoundError, ValueError, TypeError) as e:
#             print(f"An error occurred: {e}")
#         except Exception as e:
#             print(f"An unexpected error occurred: {e}")

# Clean up old functions
# def assess_image_quality(...): ...
# def check_background_complexity(...): ...
# def recommend_processing_strategy(...): ... 