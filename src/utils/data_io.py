"""
Data loading and saving utilities (metadata, images) for the pipeline.

Requires `pandas` for metadata handling.
"""

import os
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Union, Optional, Literal
import warnings
import cv2 # Needed for saving numpy arrays correctly

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    warnings.warn("Pandas library not found. Metadata loading functions will not be available.")
    pd = None

IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff']

def load_metadata(csv_path: Union[str, Path]) -> Optional['pd.DataFrame']:
    """
    Loads metadata from a CSV file into a pandas DataFrame.

    Args:
        csv_path: Path to the metadata CSV file.

    Returns:
        Optional[pd.DataFrame]: DataFrame containing the metadata, or None if loading fails.
    """
    if not PANDAS_AVAILABLE:
        warnings.warn("Pandas not installed, cannot load metadata.")
        return None
        
    try:
        path = Path(csv_path)
        if not path.is_file():
            raise FileNotFoundError(f"Metadata file not found: {path}")
        df = pd.read_csv(path)
        return df
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None
    except Exception as e:
        print(f"Error loading metadata from {csv_path}: {e}")
        return None

def get_image_paths(base_dir: Union[str, Path], recursive: bool = True) -> List[Path]:
    """
    Retrieves a list of image file paths from a directory.

    Args:
        base_dir: The directory to search for images.
        recursive (bool): If True, searches recursively through subdirectories.

    Returns:
        List[Path]: A list of Path objects for the found images.
    """
    image_paths = []
    try:
        path = Path(base_dir)
        if not path.is_dir():
            raise NotADirectoryError(f"Base directory not found or is not a directory: {path}")

        glob_pattern = "**/*" if recursive else "*"
        
        for item in path.glob(glob_pattern):
            if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS:
                image_paths.append(item)
                
    except NotADirectoryError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error scanning directory {base_dir}: {e}")
        
    return image_paths

def load_image(
    image_path: Union[str, Path],
    mode: Optional[Literal['RGB', 'L', 'RGBA']] = 'RGB' # Target PIL mode
) -> Optional[Image.Image]:
    """
    Loads an image file using Pillow.

    Args:
        image_path: Path to the image file.
        mode (Optional[Literal['RGB', 'L', 'RGBA']]): The mode to convert the image to.
            'RGB': Color image.
            'L': Grayscale image.
            'RGBA': Color image with alpha.
            Defaults to 'RGB'. If None, keeps original mode.

    Returns:
        Optional[Image.Image]: The loaded PIL Image object, or None if loading fails.
    """
    try:
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Image file not found: {path}")
        
        img = Image.open(path)
        
        # Ensure EXIF orientation is applied (if present)
        try:
             from PIL import ExifTags
             for orientation in ExifTags.TAGS.keys():
                 if ExifTags.TAGS[orientation]=='Orientation':
                     break
             exif = img._getexif()
             if exif is not None and orientation in exif:
                 exif_orientation = exif[orientation]
                 if exif_orientation == 3: img=img.rotate(180, expand=True)
                 elif exif_orientation == 6: img=img.rotate(270, expand=True)
                 elif exif_orientation == 8: img=img.rotate(90, expand=True)
        except (AttributeError, KeyError, IndexError, ImportError):
             # Cases where exif data isn't found or processing fails
             pass 
             
        if mode and img.mode != mode:
             img = img.convert(mode)
             
        return img
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def save_image(
    image_data: Union[np.ndarray, Image.Image], 
    save_path: Union[str, Path],
    quality: Optional[int] = 95 # For JPEG/WEBP
) -> bool:
    """
    Saves an image (NumPy array or PIL Image) to the specified path.

    Handles RGB, RGBA, and Grayscale images.
    Converts NumPy arrays (assumed RGB) to BGR for OpenCV saving if needed.

    Args:
        image_data: The image data to save (NumPy array or PIL.Image).
        save_path: The full path (including filename and extension) where 
                   the image will be saved.
        quality (Optional[int]): Quality setting for JPEG/WEBP formats (0-100).
                                 Ignored for lossless formats like PNG.

    Returns:
        bool: True if saving was successful, False otherwise.
    """
    try:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        
        file_suffix = path.suffix.lower()

        if isinstance(image_data, Image.Image):
            # Save PIL Image directly
            save_kwargs = {}
            if file_suffix in ['.jpg', '.jpeg', '.webp'] and quality is not None:
                 save_kwargs['quality'] = max(0, min(100, quality))
                 if file_suffix == '.webp': save_kwargs['lossless'] = False
            elif file_suffix == '.png':
                save_kwargs['compress_level'] = 6 # Default reasonable compression for PNG
            
            image_data.save(path, **save_kwargs)
            # print(f"Image saved (PIL) at: {path}")
            return True
            
        elif isinstance(image_data, np.ndarray):
            # Handle NumPy array saving
            if image_data.dtype != np.uint8:
                 # Attempt conversion if float (assume range 0-1 or 0-255)
                 if np.issubdtype(image_data.dtype, np.floating):
                      if np.max(image_data) <= 1.0 and np.min(image_data) >= 0.0:
                           image_data = (image_data * 255).astype(np.uint8)
                      else:
                           image_data = np.clip(image_data, 0, 255).astype(np.uint8)
                 else:
                      warnings.warn(f"Attempting to save NumPy array with dtype {image_data.dtype}. Converting to uint8.")
                      image_data = image_data.astype(np.uint8)
            
            save_params = []
            if file_suffix in ['.jpg', '.jpeg']: 
                save_params = [cv2.IMWRITE_JPEG_QUALITY, max(0, min(100, quality))]
            elif file_suffix == '.webp':
                 save_params = [cv2.IMWRITE_WEBP_QUALITY, max(0, min(100, quality))]
            elif file_suffix == '.png':
                 save_params = [cv2.IMWRITE_PNG_COMPRESSION, 6] # Default reasonable compression
                 
            # OpenCV expects BGR by default for color images when writing
            if image_data.ndim == 3 and image_data.shape[2] == 3: # RGB
                img_to_save = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)
            elif image_data.ndim == 3 and image_data.shape[2] == 4: # RGBA
                # OpenCV imwrite handles RGBA PNGs correctly, needs BGR(A) for others if applicable
                if file_suffix == '.png':
                    img_to_save = image_data # Save as is
                else:
                    # Convert RGBA to RGB (discard alpha) for formats like JPEG
                    warnings.warn(f"Saving RGBA image as {file_suffix}. Discarding alpha channel.")
                    img_to_save = cv2.cvtColor(image_data, cv2.COLOR_RGBA2BGR)
            elif image_data.ndim == 2: # Grayscale
                 img_to_save = image_data
            else:
                 raise ValueError(f"Unsupported NumPy array shape for saving: {image_data.shape}")
                 
            success = cv2.imwrite(str(path), img_to_save, save_params)
            if not success:
                 raise IOError(f"OpenCV failed to write image to {path}")
                 
            # print(f"Image saved (NumPy/CV2) at: {path}")
            return True
            
        else:
            raise TypeError(f"Unsupported image data type for saving: {type(image_data)}")

    except Exception as e:
        print(f"Error saving image to {save_path}: {e}")
        return False 