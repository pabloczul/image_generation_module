"""
Background image loading, combination, and simple effect utilities.
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from pathlib import Path
from typing import Union, Tuple, Optional
import warnings


def load_background_image(
    bg_input: Union[str, Path, np.ndarray, Image.Image],
    target_size: Tuple[int, int] # (width, height)
) -> Optional[np.ndarray]:
    """
    Loads a background image and resizes it to the target dimensions.

    Args:
        bg_input: Path/string to the background image file, a NumPy array (RGB/BGR),
                  or a PIL Image object.
        target_size (Tuple[int, int]): Target size (width, height).

    Returns:
        Optional[np.ndarray]: Loaded and resized background image as a NumPy array (RGB),
                            or None if loading/resizing fails.
    """
    try:
        if isinstance(bg_input, (str, Path)):
            bg_path = Path(bg_input)
            if not bg_path.is_file():
                raise FileNotFoundError(f"Background image not found: {bg_path}")
            bg_image_pil = Image.open(bg_path).convert('RGB')
        elif isinstance(bg_input, np.ndarray):
            # Assume BGR if 3 channels, convert to RGB PIL
            if bg_input.ndim == 3 and bg_input.shape[2] == 3:
                 bg_image_pil = Image.fromarray(cv2.cvtColor(bg_input.astype(np.uint8), cv2.COLOR_BGR2RGB))
            elif bg_input.ndim == 2: # Grayscale
                 bg_image_pil = Image.fromarray(bg_input.astype(np.uint8)).convert('RGB')
            else:
                 raise ValueError("Input NumPy array must be 3-channel or grayscale.")
        elif isinstance(bg_input, Image.Image):
            bg_image_pil = bg_input.convert('RGB')
        else:
            raise TypeError(f"Unsupported input type for background: {type(bg_input)}")

        # Resize using high-quality downsampling
        resized_bg = bg_image_pil.resize(target_size, Image.Resampling.LANCZOS)
        
        return np.array(resized_bg)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None
    except Exception as e:
        print(f"Error loading or resizing background image: {e}")
        return None


def combine_foreground_background(
    foreground_rgba: Union[np.ndarray, Image.Image], 
    background_rgb: Union[np.ndarray, Image.Image]
) -> Optional[np.ndarray]:
    """
    Blends an RGBA foreground image onto an RGB background image.

    Assumes foreground_rgba has an alpha channel used for blending.
    Resizes background to match foreground if dimensions differ.

    Args:
        foreground_rgba: NumPy array (H, W, 4) or PIL Image (RGBA) of the foreground.
        background_rgb: NumPy array (H, W, 3) or PIL Image (RGB) of the background.

    Returns:
        Optional[np.ndarray]: Combined image as an RGB NumPy array, or None if failed.
    """
    try:
        # Convert inputs to PIL Images for consistent handling
        if isinstance(foreground_rgba, np.ndarray):
             if foreground_rgba.shape[2] != 4:
                 raise ValueError("Foreground NumPy array must be RGBA (H, W, 4).")
             fg_pil = Image.fromarray(foreground_rgba.astype(np.uint8), 'RGBA')
        elif isinstance(foreground_rgba, Image.Image):
             fg_pil = foreground_rgba.convert('RGBA')
        else:
             raise TypeError(f"Unsupported foreground type: {type(foreground_rgba)}")

        if isinstance(background_rgb, np.ndarray):
             if background_rgb.shape[2] != 3:
                 raise ValueError("Background NumPy array must be RGB (H, W, 3).")
             # Assume input numpy is RGB, consistent with PIL output
             bg_pil = Image.fromarray(background_rgb.astype(np.uint8), 'RGB') 
        elif isinstance(background_rgb, Image.Image):
             bg_pil = background_rgb.convert('RGB')
        else:
             raise TypeError(f"Unsupported background type: {type(background_rgb)}")

        # Ensure background matches foreground size
        if fg_pil.size != bg_pil.size:
             warnings.warn(f"Background size {bg_pil.size} differs from foreground {fg_pil.size}. Resizing background.")
             bg_pil = bg_pil.resize(fg_pil.size, Image.Resampling.LANCZOS)

        # Extract alpha channel from foreground
        alpha = fg_pil.getchannel('A')

        # Composite foreground over background
        combined_pil = Image.alpha_composite(bg_pil.convert('RGBA'), fg_pil)
        
        # Return as RGB NumPy array
        return np.array(combined_pil.convert('RGB'))

    except Exception as e:
        print(f"Error combining foreground and background: {e}")
        return None
        

def add_simple_drop_shadow(
    image_rgb: Union[np.ndarray, Image.Image], 
    foreground_mask: Union[np.ndarray, Image.Image], 
    intensity: float = 0.5, 
    blur_radius: int = 15, 
    offset: Tuple[int, int] = (10, 10)
) -> Optional[np.ndarray]:
    """
    Adds a simple, offset drop shadow based on the foreground mask's bounding box.

    Note: This creates a basic rectangular shadow, not contour-aware.

    Args:
        image_rgb: The combined RGB image (NumPy array or PIL Image) 
                   onto which the shadow will be added.
        foreground_mask: Binary or grayscale mask (NumPy array or PIL Image) 
                         where non-zero indicates the foreground object.
        intensity (float): Shadow opacity (0.0 to 1.0).
        blur_radius (int): Gaussian blur radius for the shadow softness.
        offset (Tuple[int, int]): Shadow offset (x, y) relative to the object.

    Returns:
        Optional[np.ndarray]: Image (RGB NumPy array) with the added shadow, or None if failed.
    """
    try:
        # Convert image to PIL RGBA for compositing
        if isinstance(image_rgb, np.ndarray):
             img_pil_rgba = Image.fromarray(image_rgb.astype(np.uint8)).convert('RGBA')
        elif isinstance(image_rgb, Image.Image):
             img_pil_rgba = image_rgb.convert('RGBA')
        else:
             raise TypeError(f"Unsupported image type: {type(image_rgb)}")

        # Convert mask to PIL binary
        if isinstance(foreground_mask, np.ndarray):
             mask_pil = Image.fromarray(foreground_mask.astype(np.uint8)).convert('L')
        elif isinstance(foreground_mask, Image.Image):
             mask_pil = foreground_mask.convert('L')
        else:
             raise TypeError(f"Unsupported mask type: {type(foreground_mask)}")
             
        binary_mask_pil = mask_pil.point(lambda p: 255 if p > 127 else 0)

        # Get bounding box of the foreground object from the mask
        bbox = binary_mask_pil.getbbox()
        if not bbox:
             warnings.warn("Cannot add shadow: No foreground object found in mask (empty bounding box).")
             return np.array(img_pil_rgba.convert('RGB')) # Return original image

        # Create shadow layer
        shadow_layer = Image.new('RGBA', img_pil_rgba.size, (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow_layer)

        # Draw the bounding box shape offsetted
        x0, y0, x1, y1 = bbox
        shadow_color = (0, 0, 0, int(255 * max(0.0, min(1.0, intensity)))) # Clamped intensity
        shadow_draw.rectangle(
            [x0 + offset[0], y0 + offset[1], x1 + offset[0], y1 + offset[1]],
            fill=shadow_color
        )

        # Apply blur
        if blur_radius > 0:
             shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(blur_radius))

        # Composite shadow underneath the original image content
        # We need the original foreground isolated to put *over* the shadow + background
        # This function assumes `image_rgb` is already combined, making true under-shadow hard.
        # Alternative: Composite shadow *over* the combined image.
        final_image_pil = Image.alpha_composite(img_pil_rgba, shadow_layer)

        return np.array(final_image_pil.convert('RGB'))

    except Exception as e:
        print(f"Error adding drop shadow: {e}")
        return None 