# -*- coding: utf-8 -*-
"""
Tests for the background generation utilities in src.background.generators.
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

from src.background.generators import (
    generate_solid_background,
    generate_gradient_background,
    generate_textured_background,
    create_solid_background, # Also test the saving wrappers
    create_gradient_background,
    create_textured_background
)
from src.utils.data_io import load_image # For checking saved files

# --- Constants ---
TEST_WIDTH = 64
TEST_HEIGHT = 48
DEFAULT_COLOR_WHITE = (255, 255, 255)
DEFAULT_COLOR_GRAY = (128, 128, 128)
DEFAULT_GRADIENT_COLORS = [(240, 240, 240), (200, 200, 200)]

# --- Tests for generate_solid_background --- 

def test_generate_solid_default_white():
    """Test generating a solid background with default white color."""
    img = generate_solid_background(width=TEST_WIDTH, height=TEST_HEIGHT)
    assert isinstance(img, Image.Image)
    assert img.size == (TEST_WIDTH, TEST_HEIGHT)
    assert img.mode == 'RGB'
    # Check a few pixels
    assert img.getpixel((0, 0)) == DEFAULT_COLOR_WHITE
    assert img.getpixel((TEST_WIDTH // 2, TEST_HEIGHT // 2)) == DEFAULT_COLOR_WHITE

def test_generate_solid_custom_color():
    """Test generating a solid background with a custom color."""
    custom_color = (50, 100, 150)
    img = generate_solid_background(width=TEST_WIDTH, height=TEST_HEIGHT, color=custom_color)
    assert isinstance(img, Image.Image)
    assert img.size == (TEST_WIDTH, TEST_HEIGHT)
    assert img.mode == 'RGB'
    assert img.getpixel((1, 1)) == custom_color

def test_generate_solid_invalid_dims():
    """Test that invalid dimensions return None."""
    assert generate_solid_background(width=0, height=TEST_HEIGHT) is None
    assert generate_solid_background(width=TEST_WIDTH, height=-10) is None

def test_generate_solid_invalid_color():
    """Test that invalid color formats return None."""
    assert generate_solid_background(color=(255, 255)) is None # Too short
    assert generate_solid_background(color=(255, 300, 0)) is None # Value out of range (although PIL might clamp)
    assert generate_solid_background(color="blue") is None # String not supported by generate_

# --- Tests for generate_gradient_background --- 

def test_generate_gradient_defaults():
    """Test generating a gradient with default parameters."""
    img = generate_gradient_background(width=TEST_WIDTH, height=TEST_HEIGHT)
    assert isinstance(img, Image.Image)
    assert img.size == (TEST_WIDTH, TEST_HEIGHT)
    assert img.mode == 'RGB'
    # Check top and bottom pixels for default vertical gradient
    assert img.getpixel((0, 0)) == DEFAULT_GRADIENT_COLORS[0] # Top color
    # Bottom color might be slightly off due to interpolation/blur
    # assert img.getpixel((0, TEST_HEIGHT - 1)) == DEFAULT_GRADIENT_COLORS[1] 
    # Check general color range
    assert img.getpixel((TEST_WIDTH // 2, TEST_HEIGHT - 1))[0] <= DEFAULT_GRADIENT_COLORS[0][0]
    assert img.getpixel((TEST_WIDTH // 2, TEST_HEIGHT - 1))[0] >= DEFAULT_GRADIENT_COLORS[1][0]

def test_generate_gradient_horizontal():
    """Test horizontal gradient generation."""
    colors = [(255, 0, 0), (0, 0, 255)] # Red to Blue
    img = generate_gradient_background(width=TEST_WIDTH, height=TEST_HEIGHT, colors=colors, direction='horizontal')
    assert isinstance(img, Image.Image)
    assert img.size == (TEST_WIDTH, TEST_HEIGHT)
    assert img.getpixel((0, 0)) == colors[0]
    # Right side should be close to blue
    assert img.getpixel((TEST_WIDTH - 1, TEST_HEIGHT // 2))[0] <= 10 # Low red component
    assert img.getpixel((TEST_WIDTH - 1, TEST_HEIGHT // 2))[2] >= 245 # High blue component

def test_generate_gradient_radial():
    """Test radial gradient generation."""
    colors = [(255, 255, 255), (0, 0, 0)] # White center to Black edge
    img = generate_gradient_background(width=TEST_WIDTH, height=TEST_HEIGHT, colors=colors, direction='radial')
    assert isinstance(img, Image.Image)
    assert img.size == (TEST_WIDTH, TEST_HEIGHT)
    # Center pixel should be white
    assert img.getpixel((TEST_WIDTH // 2, TEST_HEIGHT // 2)) == colors[0]
    # Corner pixel should be black (or close due to blur)
    corner_pixel = img.getpixel((0, 0))
    assert sum(corner_pixel) < 30 # Sum of RGB should be low for near black

def test_generate_gradient_invalid_direction():
    """Test invalid direction raises ValueError."""
    with pytest.raises(ValueError):
        generate_gradient_background(direction='diagonal')

def test_generate_gradient_invalid_colors():
    """Test invalid colors format returns None."""
    assert generate_gradient_background(colors=[(255,0,0)]) is None # Only one color
    assert generate_gradient_background(colors="red") is None # String invalid
    assert generate_gradient_background(colors=[(255,0), (0,255)]) is None # Not RGB tuples

# --- Tests for generate_textured_background --- 

def test_generate_texture_noise():
    """Test generating noise texture."""
    img = generate_textured_background(width=TEST_WIDTH, height=TEST_HEIGHT, texture_type='noise')
    assert isinstance(img, Image.Image)
    assert img.size == (TEST_WIDTH, TEST_HEIGHT)
    assert img.mode == 'RGB'
    # Noise is random, hard to assert specific pixels, check general range and variance
    img_array = np.array(img)
    assert np.mean(img_array) > 190 # Should be light overall
    assert np.std(img_array) > 5 # Should have some variance

def test_generate_texture_canvas():
    """Test generating canvas texture."""
    img = generate_textured_background(width=TEST_WIDTH, height=TEST_HEIGHT, texture_type='canvas')
    assert isinstance(img, Image.Image)
    assert img.size == (TEST_WIDTH, TEST_HEIGHT)
    assert img.mode == 'RGB'
    # Canvas is mostly base color with slight variations
    base_color = (245, 245, 240)
    assert abs(np.mean(np.array(img)[:,:,0]) - base_color[0]) < 10

def test_generate_texture_invalid_type():
    """Test invalid texture type raises ValueError."""
    with pytest.raises(ValueError):
        generate_textured_background(texture_type='wood')

# --- Tests for create_* saving wrappers ---

def test_create_solid_background_saves(tmp_path):
    """Test that the create_solid_background wrapper saves a file."""
    filename = "create_solid_test.png"
    success = create_solid_background(filename=filename, output_dir=tmp_path, 
                                      width=TEST_WIDTH, height=TEST_HEIGHT, color=(0, 255, 0))
    assert success is True
    saved_path = tmp_path / filename
    assert saved_path.is_file()
    # Load and verify
    loaded_img = load_image(saved_path)
    assert loaded_img.size == (TEST_WIDTH, TEST_HEIGHT)
    assert loaded_img.getpixel((0,0)) == (0, 255, 0)

def test_create_gradient_background_saves(tmp_path):
    """Test that the create_gradient_background wrapper saves a file."""
    filename = "create_gradient_test.jpg"
    colors = [(0, 0, 0), (255, 255, 255)] # Black to white
    success = create_gradient_background(filename=filename, output_dir=tmp_path, 
                                         width=TEST_WIDTH, height=TEST_HEIGHT, 
                                         colors=colors, direction='horizontal')
    assert success is True
    saved_path = tmp_path / filename
    assert saved_path.is_file()
    # Load and verify basic properties (JPEG is lossy)
    loaded_img = load_image(saved_path)
    assert loaded_img.size == (TEST_WIDTH, TEST_HEIGHT)
    assert loaded_img.mode == 'RGB'

def test_create_textured_background_saves(tmp_path):
    """Test that the create_textured_background wrapper saves a file."""
    filename = "create_texture_test.png"
    success = create_textured_background(filename=filename, output_dir=tmp_path, 
                                         width=TEST_WIDTH, height=TEST_HEIGHT, 
                                         texture_type='canvas')
    assert success is True
    saved_path = tmp_path / filename
    assert saved_path.is_file()
    loaded_img = load_image(saved_path)
    assert loaded_img.size == (TEST_WIDTH, TEST_HEIGHT)
    assert loaded_img.mode == 'RGB'

def test_create_overwrite_false(tmp_path):
    """Test that overwrite=False skips saving if file exists."""
    filename = "overwrite_test.png"
    path = tmp_path / filename
    path.touch() # Create dummy existing file
    
    success = create_solid_background(filename, output_dir=tmp_path, overwrite=False)
    assert success is True # Should return True as it 'already exists'
    assert path.read_text() == "" # Check if content is unchanged (still empty)

def test_create_overwrite_true(tmp_path):
    """Test that overwrite=True saves even if file exists."""
    filename = "overwrite_test.png"
    path = tmp_path / filename
    path.write_text("old content") # Create dummy existing file
    
    success = create_solid_background(filename, output_dir=tmp_path, 
                                      width=10, height=10, color=(0,0,0), overwrite=True)
    assert success is True
    assert path.is_file()
    assert path.stat().st_size > 0 # Check if file has actual image data now
    assert path.read_text() != "old content" 