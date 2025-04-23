# -*- coding: utf-8 -*-
"""
Tests for the data I/O utilities in src.utils.data_io.
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

from src.utils.data_io import load_image, save_image, get_image_paths

# --- Helper function to create dummy images ---

def create_dummy_image(width: int, height: int, fmt: str = 'png', 
                       mode: str = 'RGB', color=(255, 0, 0)) -> Path:
    """Creates a simple dummy image file for testing.
       Requires a place to save it, ideally use tmp_path fixture.
    """
    img = Image.new(mode, (width, height), color=color)
    # Need a temporary path provided by pytest fixture
    raise NotImplementedError("This helper needs tmp_path to save the image.")

# --- Tests for load_image --- 

def test_load_image_success_rgb(tmp_path):
    """Test loading a valid RGB image returns a PIL Image in RGB mode."""
    # Create a dummy PNG image
    img_path = tmp_path / "test_rgb.png"
    dummy_img = Image.new('RGB', (10, 20), color='red')
    dummy_img.save(img_path)

    loaded_img = load_image(img_path, mode='RGB')
    assert isinstance(loaded_img, Image.Image)
    assert loaded_img.mode == 'RGB'
    assert loaded_img.size == (10, 20)

def test_load_image_success_grayscale(tmp_path):
    """Test loading a valid grayscale image returns a PIL Image in L mode."""
    img_path = tmp_path / "test_gray.png"
    dummy_img = Image.new('L', (15, 5), color=128)
    dummy_img.save(img_path)

    loaded_img = load_image(img_path, mode='L')
    assert isinstance(loaded_img, Image.Image)
    assert loaded_img.mode == 'L'
    assert loaded_img.size == (15, 5)

def test_load_image_mode_conversion(tmp_path):
    """Test loading an image and converting its mode."""
    img_path = tmp_path / "test_mode_conv.png"
    dummy_img = Image.new('L', (8, 8), color=100)
    dummy_img.save(img_path)

    # Load as RGB
    loaded_img = load_image(img_path, mode='RGB')
    assert isinstance(loaded_img, Image.Image)
    assert loaded_img.mode == 'RGB'
    assert loaded_img.size == (8, 8)
    # Check if pixel values look reasonable (grayscale converted to RGB)
    assert loaded_img.getpixel((0, 0)) == (100, 100, 100)

def test_load_image_non_existent():
    """Test loading a non-existent file returns None."""
    non_existent_path = Path("non/existent/path/image.jpg")
    loaded_img = load_image(non_existent_path)
    assert loaded_img is None

def test_load_image_invalid_file(tmp_path):
    """Test loading an invalid (non-image) file returns None."""
    invalid_file = tmp_path / "not_an_image.txt"
    invalid_file.write_text("this is not an image")
    loaded_img = load_image(invalid_file)
    assert loaded_img is None

# --- Tests for save_image --- 

def test_save_image_pil(tmp_path):
    """Test saving a PIL image successfully creates a file."""
    img_to_save = Image.new('RGB', (5, 5), color='blue')
    save_path_png = tmp_path / "pil_save_test.png"
    save_path_jpg = tmp_path / "pil_save_test.jpg"

    success_png = save_image(img_to_save, save_path_png)
    assert success_png is True
    assert save_path_png.is_file()
    # Optionally load back and check
    loaded_png = Image.open(save_path_png)
    assert loaded_png.size == (5, 5)
    assert loaded_png.mode == 'RGB' # PNG saves RGB directly

    success_jpg = save_image(img_to_save, save_path_jpg, quality=80)
    assert success_jpg is True
    assert save_path_jpg.is_file()
    loaded_jpg = Image.open(save_path_jpg)
    assert loaded_jpg.size == (5, 5)
    # JPEG might slightly alter colors


def test_save_image_numpy_rgb(tmp_path):
    """Test saving an RGB NumPy array."""
    img_array = np.zeros((10, 12, 3), dtype=np.uint8)
    img_array[:, :, 1] = 255 # Make it green
    save_path_png = tmp_path / "np_save_test.png"
    save_path_jpg = tmp_path / "np_save_test.jpg"

    success_png = save_image(img_array, save_path_png)
    assert success_png is True
    assert save_path_png.is_file()
    loaded_png = Image.open(save_path_png)
    assert loaded_png.size == (12, 10) # Note PIL size is (width, height)
    assert loaded_png.mode == 'RGB'
    assert loaded_png.getpixel((0,0)) == (0, 255, 0)

    success_jpg = save_image(img_array, save_path_jpg, quality=90)
    assert success_jpg is True
    assert save_path_jpg.is_file()
    loaded_jpg = Image.open(save_path_jpg)
    assert loaded_jpg.size == (12, 10)

def test_save_image_numpy_grayscale(tmp_path):
    """Test saving a Grayscale NumPy array."""
    img_array = np.ones((6, 8), dtype=np.uint8) * 150
    save_path = tmp_path / "np_gray_save.png"

    success = save_image(img_array, save_path)
    assert success is True
    assert save_path.is_file()
    loaded = Image.open(save_path)
    assert loaded.size == (8, 6)
    assert loaded.mode == 'L' # Should save as grayscale
    assert loaded.getpixel((0,0)) == 150

def test_save_image_invalid_type(tmp_path):
    """Test saving an invalid data type returns False."""
    invalid_data = "this is not an image"
    save_path = tmp_path / "invalid_save.png"
    success = save_image(invalid_data, save_path)
    assert success is False
    assert not save_path.exists()

# --- Tests for get_image_paths (Basic) ---

def test_get_image_paths_empty(tmp_path):
    """Test getting paths from an empty directory."""
    assert get_image_paths(tmp_path) == []

def test_get_image_paths_flat(tmp_path):
    """Test getting paths from a directory with images and other files."""
    (tmp_path / "img1.jpg").touch()
    (tmp_path / "img2.PNG").touch()
    (tmp_path / "img3.jpeg").touch()
    (tmp_path / "readme.txt").touch()
    (tmp_path / "archive.zip").touch()
    
    expected_paths = sorted([
        tmp_path / "img1.jpg",
        tmp_path / "img2.PNG",
        tmp_path / "img3.jpeg"
    ])
    actual_paths = sorted(get_image_paths(tmp_path, recursive=False))
    assert actual_paths == expected_paths

def test_get_image_paths_recursive(tmp_path):
    """Test recursive path finding."""
    sub_dir = tmp_path / "subdir"
    sub_sub_dir = sub_dir / "subsubdir"
    sub_sub_dir.mkdir(parents=True)

    (tmp_path / "root.png").touch()
    (sub_dir / "sub.webp").touch()
    (sub_sub_dir / "subsub.gif").touch() # gif is not in default extensions
    (sub_sub_dir / "subsub.tiff").touch()
    (sub_dir / "notes.md").touch()
    
    expected_paths = sorted([
        tmp_path / "root.png",
        sub_dir / "sub.webp",
        sub_sub_dir / "subsub.tiff"
    ])
    actual_paths = sorted(get_image_paths(tmp_path, recursive=True))
    assert actual_paths == expected_paths

def test_get_image_paths_non_recursive(tmp_path):
    """Test non-recursive path finding."""
    sub_dir = tmp_path / "subdir"
    sub_dir.mkdir()
    (tmp_path / "root.png").touch()
    (sub_dir / "sub.jpeg").touch()

    expected_paths = [tmp_path / "root.png"]
    actual_paths = get_image_paths(tmp_path, recursive=False)
    assert actual_paths == expected_paths

def test_get_image_paths_non_existent_dir():
    """Test getting paths from a non-existent directory."""
    assert get_image_paths("non/existent/dir") == [] 