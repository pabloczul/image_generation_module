#!/usr/bin/env python3
"""
Module for generating synthetic background images (gradients, solids, textures).
"""

from PIL import Image, ImageDraw, ImageFilter
from pathlib import Path
import numpy as np
import random
from typing import Tuple, List, Optional, Union
import time # Added for generate_standard_backgrounds timing

# Import save_image utility
from ..utils.data_io import save_image

# Default configuration (Consider moving to a central config file)
DEFAULT_OUTPUT_DIR = Path("../../data/backgrounds") # Relative to this file's location
DEFAULT_WIDTH = 1024 # Reduced default size
DEFAULT_HEIGHT = 1024

def generate_gradient_background(
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    colors: Optional[List[Tuple[int, int, int]]] = None,
    direction: str = 'vertical',
) -> Optional[Image.Image]:
    """
    Generates a gradient background image as a PIL Image object.

    Args:
        width (int): Width of the background.
        height (int): Height of the background.
        colors (Optional[List[Tuple[int, int, int]]]): List of 2 RGB color tuples for the gradient.
                                                       Defaults to a light gray gradient.
        direction (str): Direction of gradient ('vertical', 'horizontal', 'radial').

    Returns:
        Optional[Image.Image]: The generated gradient PIL Image, or None if error.
    """
    if colors is None:
        colors = [(245, 245, 245), (230, 230, 230)]  # Default: Lighter gray gradient
    if not isinstance(colors, list) or len(colors) < 2 or not all(isinstance(c, tuple) and len(c)==3 for c in colors):
        print("Error: Invalid colors format. Expected list of two RGB tuples.")
        return None
    if width <= 0 or height <= 0:
        print(f"Error: Invalid dimensions ({width}x{height}). Must be positive.")
        return None

    try:
        image = Image.new('RGB', (width, height), color=colors[0])
        draw = ImageDraw.Draw(image)

        c1 = colors[0]
        c2 = colors[1]

        if direction == 'vertical':
            for y in range(height):
                ratio = y / (height - 1) if height > 1 else 0
                r = int(c1[0] * (1 - ratio) + c2[0] * ratio)
                g = int(c1[1] * (1 - ratio) + c2[1] * ratio)
                b = int(c1[2] * (1 - ratio) + c2[2] * ratio)
                draw.line([(0, y), (width, y)], fill=(r, g, b))

        elif direction == 'horizontal':
            for x in range(width):
                ratio = x / (width - 1) if width > 1 else 0
                r = int(c1[0] * (1 - ratio) + c2[0] * ratio)
                g = int(c1[1] * (1 - ratio) + c2[1] * ratio)
                b = int(c1[2] * (1 - ratio) + c2[2] * ratio)
                draw.line([(x, 0), (x, height)], fill=(r, g, b))

        elif direction == 'radial':
            center_x, center_y = width // 2, height // 2
            max_dist = np.sqrt(center_x**2 + center_y**2)
            if max_dist == 0: max_dist = 1 # Avoid division by zero

            for y in range(height):
                for x in range(width):
                    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                    ratio = min(1.0, dist / max_dist)
                    r = int(c1[0] * (1 - ratio) + c2[0] * ratio)
                    g = int(c1[1] * (1 - ratio) + c2[1] * ratio)
                    b = int(c1[2] * (1 - ratio) + c2[2] * ratio)
                    image.putpixel((x, y), (r, g, b)) # Use putpixel for radial
        else:
            raise ValueError(f"Invalid gradient direction: {direction}. Choose 'vertical', 'horizontal', or 'radial'.")

        # Apply slight blur to smooth the gradient
        image = image.filter(ImageFilter.GaussianBlur(radius=1))
        return image

    except Exception as e:
        print(f"Error generating gradient background: {e}")
        return None

def generate_solid_background(
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    color: Optional[Tuple[int, int, int]] = None,
) -> Optional[Image.Image]:
    """
    Generates a solid color background image as a PIL Image object.

    Args:
        width (int): Width of the background.
        height (int): Height of the background.
        color (Optional[Tuple[int, int, int]]): RGB color tuple. Defaults to white.

    Returns:
        Optional[Image.Image]: The generated solid color PIL Image, or None if error.
    """
    if color is None:
        color = (255, 255, 255)  # Default: White
    if not isinstance(color, tuple) or len(color) != 3:
        print("Error: Invalid color format. Expected RGB tuple.")
        return None
    if width <= 0 or height <= 0:
        print(f"Error: Invalid dimensions ({width}x{height}). Must be positive.")
        return None
        
    try:
        image = Image.new('RGB', (width, height), color=color)
        return image
    except Exception as e:
        print(f"Error generating solid background: {e}")
        return None

def generate_textured_background(
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    texture_type: str = 'noise',
) -> Optional[Image.Image]:
    """
    Generates a textured background image as a PIL Image object.

    Args:
        width (int): Width of the background.
        height (int): Height of the background.
        texture_type (str): Type of texture ('noise', 'canvas').

    Returns:
        Optional[Image.Image]: The generated textured PIL Image, or None if error.
    """
    if width <= 0 or height <= 0:
        print(f"Error: Invalid dimensions ({width}x{height}). Must be positive.")
        return None
        
    try:
        image: Optional[Image.Image] = None
        if texture_type == 'noise':
            pixels = np.random.randint(200, 256, (height, width, 3), dtype=np.uint8)
            image = Image.fromarray(pixels, 'RGB')
            image = image.filter(ImageFilter.GaussianBlur(radius=1))

        elif texture_type == 'canvas':
            base_color = (245, 245, 240)
            image = Image.new('RGB', (width, height), color=base_color)
            draw = ImageDraw.Draw(image)
            num_dots = max(1, width * height // 100) # Ensure at least 1 dot
            for _ in range(num_dots):
                x = random.randint(0, width - 1)
                y = random.randint(0, height - 1)
                color_variation = random.randint(-10, 10)
                dot_color = tuple(max(0, min(255, c + color_variation)) for c in base_color)
                draw.point((x, y), fill=dot_color)
            image = image.filter(ImageFilter.GaussianBlur(radius=0.5))

        else:
            raise ValueError(f"Invalid texture type: {texture_type}. Choose 'noise' or 'canvas'.")
        
        return image
        
    except Exception as e:
        print(f"Error generating textured background ({texture_type}): {e}")
        return None

# --- Functions that also save (can be used directly or by the pipeline) ---

def create_gradient_background(
    filename: Union[str, Path], # Now includes path
    output_dir: Optional[Union[str, Path]] = None,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    colors: Optional[List[Tuple[int, int, int]]] = None,
    direction: str = 'vertical',
    overwrite: bool = False
) -> bool:
    """
    Generates a gradient background image and saves it to a file.

    Args:
        filename: Base filename (e.g., 'my_gradient.png').
        output_dir: Directory to save the image. If None, saves next to filename.
        width, height, colors, direction: Parameters for generation.
        overwrite (bool): If True, overwrite existing file.

    Returns:
        bool: True if image generation and saving were successful.
    """
    path = Path(filename) if output_dir is None else Path(output_dir) / filename
    
    if not overwrite and path.exists():
        print(f"✓ Already exists (gradient): {path}")
        return True
        
    image = generate_gradient_background(width=width, height=height, colors=colors, direction=direction)
    
    if image is None:
        print(f"✗ Failed to generate gradient background for {path}")
        return False
        
    print(f"Saving gradient background to: {path}...")
    if save_image(image, path):
        print(f"✓ Saved gradient background: {path}")
        return True
    else:
        print(f"✗ Failed to save gradient background {path}")
        return False
        

def create_solid_background(
    filename: Union[str, Path], 
    output_dir: Optional[Union[str, Path]] = None,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    color: Optional[Tuple[int, int, int]] = None,
    overwrite: bool = False
) -> bool:
    """
    Generates a solid color background image and saves it.

    Args:
        filename: Base filename (e.g., 'solid_white.png').
        output_dir: Directory to save the image. If None, saves next to filename.
        width, height, color: Parameters for generation.
        overwrite (bool): If True, overwrite existing file.

    Returns:
        bool: True if image generation and saving were successful.
    """
    path = Path(filename) if output_dir is None else Path(output_dir) / filename
    
    if not overwrite and path.exists():
        print(f"✓ Already exists (solid): {path}")
        return True
        
    image = generate_solid_background(width=width, height=height, color=color)
    
    if image is None:
        print(f"✗ Failed to generate solid background for {path}")
        return False
        
    print(f"Saving solid background to: {path}...")
    if save_image(image, path):
        print(f"✓ Saved solid background: {path}")
        return True
    else:
        print(f"✗ Failed to save solid background {path}")
        return False

def create_textured_background(
    filename: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    texture_type: str = 'noise',
    overwrite: bool = False
) -> bool:
    """
    Generates a textured background image and saves it.

    Args:
        filename: Base filename (e.g., 'texture_noise.png').
        output_dir: Directory to save the image. If None, saves next to filename.
        width, height, texture_type: Parameters for generation.
        overwrite (bool): If True, overwrite existing file.

    Returns:
        bool: True if image generation and saving were successful.
    """
    path = Path(filename) if output_dir is None else Path(output_dir) / filename
    
    if not overwrite and path.exists():
        print(f"✓ Already exists (texture): {path}")
        return True
        
    image = generate_textured_background(width=width, height=height, texture_type=texture_type)
    
    if image is None:
        print(f"✗ Failed to generate textured background ({texture_type}) for {path}")
        return False
        
    print(f"Saving textured background to: {path}...")
    if save_image(image, path):
        print(f"✓ Saved textured background: {path}")
        return True
    else:
        print(f"✗ Failed to save textured background {path}")
        return False

# --- Utility to generate a standard set (still useful) ---

def generate_standard_backgrounds(output_dir: Union[str, Path], overwrite: bool = False):
    """
    Generates and saves a standard set of background images to the output directory.

    Args:
        output_dir (Union[str, Path]): Directory where backgrounds will be saved.
        overwrite (bool): If True, overwrite existing files.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating standard set of backgrounds in {output_path.resolve()}...")
    width, height = DEFAULT_WIDTH, DEFAULT_HEIGHT # Use default size for standard set
    generated_count = 0
    failed_count = 0
    skipped_count = 0
    start_time = time.time()

    # Define standard sets
    gradients = [
        ("gradient_gray_vertical.jpg", "vertical", [(245, 245, 245), (230, 230, 230)]),
        ("gradient_gray_horizontal.jpg", "horizontal", [(245, 245, 245), (230, 230, 230)]),
        ("gradient_gray_radial.jpg", "radial", [(245, 245, 245), (230, 230, 230)]),
        ("gradient_blue_vertical.jpg", "vertical", [(240, 245, 255), (220, 230, 250)]),
        ("gradient_beige_radial.jpg", "radial", [(255, 250, 240), (245, 235, 220)]),
        ("gradient_pink_vertical.jpg", "vertical", [(255, 240, 245), (245, 220, 230)]),
    ]
    solids = [
        ("solid_white.jpg", (255, 255, 255)),
        ("solid_light_gray.jpg", (240, 240, 240)),
        ("solid_light_blue.jpg", (240, 245, 255)),
        ("solid_light_pink.jpg", (255, 240, 245)),
        ("solid_light_beige.jpg", (255, 250, 240)),
    ]
    textures = [
        ("texture_noise.jpg", "noise"),
        ("texture_canvas.jpg", "canvas"),
    ]

    # Process Gradients
    for filename, direction, colors in gradients:
        file_path = output_path / filename
        if not overwrite and file_path.exists(): skipped_count += 1; continue
        if create_gradient_background(filename, output_path, width, height, colors, direction, overwrite):
             generated_count += 1
        else:
             failed_count += 1

    # Process Solids
    for filename, color in solids:
        file_path = output_path / filename
        if not overwrite and file_path.exists(): skipped_count += 1; continue
        if create_solid_background(filename, output_path, width, height, color, overwrite):
            generated_count += 1
        else:
            failed_count += 1

    # Process Textures
    for filename, texture_type in textures:
        file_path = output_path / filename
        if not overwrite and file_path.exists(): skipped_count += 1; continue
        if create_textured_background(filename, output_path, width, height, texture_type, overwrite):
            generated_count += 1
        else:
            failed_count += 1

    elapsed_time = time.time() - start_time
    print("Standard Background Generation Summary:")
    print(f"  Generated/Overwritten: {generated_count}")
    print(f"  Skipped (already exist): {skipped_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Time elapsed: {elapsed_time:.2f} seconds")

# Example usage 
# if __name__ == '__main__':
#     # Generate the standard set in a test directory
#     test_dir = Path("./generated_backgrounds_test")
#     generate_standard_backgrounds(test_dir, overwrite=True)
#     print(f"\nCheck the contents of {test_dir.resolve()}")
# 
#     # Generate a single background in memory
#     print("\nGenerating a single gradient in memory...")
#     single_gradient = generate_gradient_background(width=512, height=512, colors=[(0,0,255), (0,255,255)], direction='horizontal')
#     if single_gradient:
#         print("Successfully generated gradient PIL Image.")
#         # single_gradient.show() # Uncomment to display if possible
#         save_image(single_gradient, test_dir / "single_gradient_test.png")
#     else:
#         print("Failed to generate single gradient.") 