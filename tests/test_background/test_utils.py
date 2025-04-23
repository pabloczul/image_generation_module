# -*- coding: utf-8 -*-
"""
Tests for background utility functions in src.background.utils.
"""
import sys
import pytest
from pathlib import Path
import numpy as np
from PIL import Image

# Ensure src is importable
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.background.utils import (
    load_background_image,
    combine_foreground_background,
    add_simple_drop_shadow
)

# --- Constants & Setup ---
TEST_WIDTH = 64
TEST_HEIGHT = 48

@pytest.fixture
def dummy_rgb_image() -> Image.Image:
    """Provides a simple RGB PIL Image."""
    return Image.new('RGB', (TEST_WIDTH, TEST_HEIGHT), color=(10, 20, 30))

@pytest.fixture
def dummy_rgba_image() -> Image.Image:
    """Provides a simple RGBA PIL Image with varying alpha."""
    img = Image.new('RGBA', (TEST_WIDTH, TEST_HEIGHT), color=(50, 100, 150, 255))
    # Make some pixels transparent
    for x in range(TEST_WIDTH // 2):
        img.putpixel((x, 0), (50, 100, 150, 0))
    return img
    
@pytest.fixture
def dummy_mask() -> np.ndarray:
    """Provides a simple numpy mask (e.g., foreground object)."""
    mask = np.zeros((TEST_HEIGHT, TEST_WIDTH), dtype=np.uint8)
    # Create a rectangle in the middle
    mask[TEST_HEIGHT//4 : 3*TEST_HEIGHT//4, TEST_WIDTH//4 : 3*TEST_WIDTH//4] = 255
    return mask

# --- Tests for load_background_image --- 

def test_load_background_from_path(tmp_path, dummy_rgb_image):
    """Test loading a background image from a file path."""
    bg_path = tmp_path / "bg_test.png"
    dummy_rgb_image.save(bg_path)
    target_size = (TEST_WIDTH // 2, TEST_HEIGHT // 2)
    
    loaded_bg = load_background_image(bg_path, target_size=target_size)
    assert isinstance(loaded_bg, np.ndarray)
    assert loaded_bg.shape == (target_size[1], target_size[0], 3)
    assert loaded_bg.dtype == np.uint8

def test_load_background_from_pil(dummy_rgb_image):
    """Test loading from a PIL image object."""
    target_size = (TEST_WIDTH * 2, TEST_HEIGHT * 2) # Test resizing up
    loaded_bg = load_background_image(dummy_rgb_image, target_size=target_size)
    assert isinstance(loaded_bg, np.ndarray)
    assert loaded_bg.shape == (target_size[1], target_size[0], 3)

def test_load_background_from_numpy(dummy_rgb_image):
    """Test loading from a NumPy array (simulating BGR or RGB)."""
    target_size = (TEST_WIDTH, TEST_HEIGHT)
    rgb_array = np.array(dummy_rgb_image)
    # Simulate BGR 
    # bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR) # Requires cv2 import
    bgr_array = rgb_array[:, :, ::-1] # Manual BGR swap
    
    loaded_bg_rgb = load_background_image(rgb_array, target_size=target_size)
    loaded_bg_bgr = load_background_image(bgr_array, target_size=target_size)
    
    assert isinstance(loaded_bg_rgb, np.ndarray)
    assert loaded_bg_rgb.shape == (TEST_HEIGHT, TEST_WIDTH, 3)
    assert isinstance(loaded_bg_bgr, np.ndarray)
    assert loaded_bg_bgr.shape == (TEST_HEIGHT, TEST_WIDTH, 3)
    # Check if BGR was correctly converted back to RGB numpy array
    assert np.array_equal(loaded_bg_rgb, rgb_array)
    assert np.array_equal(loaded_bg_bgr, rgb_array) 

def test_load_background_non_existent():
    """Test loading non-existent path returns None."""
    assert load_background_image("non/existent_bg.jpg", (10, 10)) is None

def test_load_background_invalid_type():
    """Test loading invalid type returns None."""
    assert load_background_image([1, 2, 3], (10, 10)) is None

# --- Tests for combine_foreground_background --- 

def test_combine_numpy_inputs(dummy_rgba_image, dummy_rgb_image):
    """Test combining two NumPy arrays (RGBA fg, RGB bg)."""
    fg_rgba_np = np.array(dummy_rgba_image)
    bg_rgb_np = np.array(dummy_rgb_image)
    
    combined = combine_foreground_background(fg_rgba_np, bg_rgb_np)
    assert isinstance(combined, np.ndarray)
    assert combined.shape == (TEST_HEIGHT, TEST_WIDTH, 3) # Should be RGB output
    assert combined.dtype == np.uint8
    # Check a pixel that was transparent in foreground - should be background color
    assert np.array_equal(combined[0, 0], bg_rgb_np[0, 0])
    # Check a pixel that was opaque in foreground - should be foreground color
    assert np.array_equal(combined[TEST_HEIGHT // 2, TEST_WIDTH // 2], fg_rgba_np[TEST_HEIGHT // 2, TEST_WIDTH // 2, :3])

def test_combine_pil_inputs(dummy_rgba_image, dummy_rgb_image):
    """Test combining two PIL Images."""
    combined = combine_foreground_background(dummy_rgba_image, dummy_rgb_image)
    assert isinstance(combined, np.ndarray) # Function returns numpy array
    assert combined.shape == (TEST_HEIGHT, TEST_WIDTH, 3)
    assert combined.dtype == np.uint8
    # Check pixels like above
    bg_rgb_np = np.array(dummy_rgb_image)
    fg_rgba_np = np.array(dummy_rgba_image)
    assert np.array_equal(combined[0, 0], bg_rgb_np[0, 0])
    assert np.array_equal(combined[TEST_HEIGHT // 2, TEST_WIDTH // 2], fg_rgba_np[TEST_HEIGHT // 2, TEST_WIDTH // 2, :3])

def test_combine_mixed_inputs(dummy_rgba_image, dummy_rgb_image):
    """Test combining mixed PIL/NumPy inputs."""
    fg_rgba_np = np.array(dummy_rgba_image)
    bg_pil = dummy_rgb_image
    combined = combine_foreground_background(fg_rgba_np, bg_pil)
    assert isinstance(combined, np.ndarray)
    assert combined.shape == (TEST_HEIGHT, TEST_WIDTH, 3)

def test_combine_size_mismatch(dummy_rgba_image, dummy_rgb_image):
    """Test background resizing when sizes mismatch."""
    bg_small_pil = dummy_rgb_image.resize((TEST_WIDTH // 2, TEST_HEIGHT // 2))
    combined = combine_foreground_background(dummy_rgba_image, bg_small_pil)
    assert isinstance(combined, np.ndarray)
    # Output shape should match the foreground image shape
    assert combined.shape == (TEST_HEIGHT, TEST_WIDTH, 3)

def test_combine_invalid_inputs(dummy_rgb_image):
    """Test invalid input types/shapes."""
    rgb_np = np.array(dummy_rgb_image)
    assert combine_foreground_background(rgb_np, rgb_np) is None # Foreground not RGBA
    assert combine_foreground_background(dummy_rgba_image, "not an image") is None

# --- Tests for add_simple_drop_shadow --- 

def test_add_shadow_numpy(dummy_rgb_image, dummy_mask):
    """Test adding shadow to a numpy image."""
    img_np = np.array(dummy_rgb_image)
    shadowed = add_simple_drop_shadow(img_np, dummy_mask, intensity=0.8, blur_radius=5, offset=(5,5))
    assert isinstance(shadowed, np.ndarray)
    assert shadowed.shape == img_np.shape
    # Check that pixels outside the mask+offset+blur area are unchanged
    # This is tricky due to blur. Check a corner pixel far from mask.
    assert np.array_equal(shadowed[0, 0], img_np[0, 0]) 
    # Check that some pixels are darker (where shadow should be)
    # Sample within the shadow offset area but outside the original mask
    shadow_check_y = TEST_HEIGHT // 4 + 5 # Inside mask Y + offset Y
    shadow_check_x = TEST_WIDTH // 4 + 5 # Inside mask X + offset X
    original_pixel_sum = np.sum(img_np[shadow_check_y, shadow_check_x])
    shadowed_pixel_sum = np.sum(shadowed[shadow_check_y, shadow_check_x])
    assert shadowed_pixel_sum < original_pixel_sum

def test_add_shadow_pil(dummy_rgb_image, dummy_mask):
    """Test adding shadow to a PIL image."""
    shadowed = add_simple_drop_shadow(dummy_rgb_image, Image.fromarray(dummy_mask))
    assert isinstance(shadowed, np.ndarray)
    assert shadowed.shape == (TEST_HEIGHT, TEST_WIDTH, 3)

def test_add_shadow_no_mask(dummy_rgb_image):
    """Test adding shadow with an empty mask returns original."""
    img_np = np.array(dummy_rgb_image)
    empty_mask = np.zeros((TEST_HEIGHT, TEST_WIDTH), dtype=np.uint8)
    shadowed = add_simple_drop_shadow(img_np, empty_mask)
    assert isinstance(shadowed, np.ndarray)
    assert np.array_equal(shadowed, img_np) # Should return the original image 