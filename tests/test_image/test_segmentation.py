# -*- coding: utf-8 -*-
"""
Tests for the image segmentation utilities in src.image.segmentation.
"""
import sys
import pytest
from pathlib import Path
import numpy as np
from PIL import Image
from unittest.mock import patch, MagicMock

# Ensure src is importable
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.image.segmentation import Segmenter, SUPPORTED_MODELS

# --- Constants & Fixtures ---
TEST_WIDTH = 64
TEST_HEIGHT = 48

@pytest.fixture
def dummy_mask_array() -> np.ndarray:
    """Provides a simple binary numpy mask (square in middle)."""
    mask = np.zeros((TEST_HEIGHT, TEST_WIDTH), dtype=np.uint8)
    mask[10:TEST_HEIGHT-10, 10:TEST_WIDTH-10] = 255
    return mask

@pytest.fixture
def noisy_mask_array() -> np.ndarray:
    """Provides a mask with small holes and noise."""
    mask = np.ones((TEST_HEIGHT, TEST_WIDTH), dtype=np.uint8) * 255
    # Add some holes (black spots)
    mask[TEST_HEIGHT//2 - 2 : TEST_HEIGHT//2 + 2, TEST_WIDTH//2 - 2 : TEST_WIDTH//2 + 2] = 0
    mask[5, 5] = 0
    mask[TEST_HEIGHT-5, TEST_WIDTH-5] = 0
    # Add noise (white spots)
    mask[0, 0] = 255 
    mask[1, 0] = 0 # Ensure some background remains for complexity calc
    return mask

@pytest.fixture
def segmenter_instance() -> Segmenter:
    """Provides a default Segmenter instance."""
    # Mock the session creation during testing to avoid actual model loads unless intended
    with patch('rembg.new_session', return_value=MagicMock()) as mock_new_session:
        instance = Segmenter() 
        mock_new_session.assert_called_once_with(model_name='u2net')
        return instance

# --- Tests for Segmenter Initialization ---

def test_segmenter_init_default():
    with patch('rembg.new_session', return_value=MagicMock()) as mock_new_session:
        segmenter = Segmenter()
        assert segmenter.model_name == 'u2net'
        assert isinstance(segmenter.session, MagicMock)
        mock_new_session.assert_called_once()

def test_segmenter_init_custom_model():
    # Assumes 'u2netp' is a supported model key
    if 'u2netp' in SUPPORTED_MODELS:
        with patch('rembg.new_session', return_value=MagicMock()) as mock_new_session:
            segmenter = Segmenter(model_name='u2netp')
            assert segmenter.model_name == 'u2netp'
            mock_new_session.assert_called_once_with(model_name='u2netp')

def test_segmenter_init_invalid_model():
    with pytest.raises(ValueError):
        Segmenter(model_name="invalid_model_name")

# --- Tests for Segmenter.refine_mask --- 

def test_refine_mask_default_ops(segmenter_instance, noisy_mask_array):
    """Test default refinement (close then dilate) fills small holes."""
    original_hole_pixel = noisy_mask_array[TEST_HEIGHT//2, TEST_WIDTH//2]
    assert original_hole_pixel == 0 # Verify hole exists
    
    refined = segmenter_instance.refine_mask(noisy_mask_array)
    assert isinstance(refined, np.ndarray)
    assert refined.shape == noisy_mask_array.shape
    assert refined.dtype == np.uint8
    # Default ops should fill the small central hole
    assert refined[TEST_HEIGHT//2, TEST_WIDTH//2] > 0 

def test_refine_mask_specific_ops(segmenter_instance, dummy_mask_array):
    """Test specific morphology operations."""
    # Erosion should shrink the mask
    erode_ops = [{'type': 'erode', 'kernel_size': 5, 'iterations': 1}]
    eroded = segmenter_instance.refine_mask(dummy_mask_array, operations=erode_ops)
    assert np.sum(eroded > 0) < np.sum(dummy_mask_array > 0)

    # Dilation should expand the mask
    dilate_ops = [{'type': 'dilate', 'kernel_size': 5, 'iterations': 1}]
    dilated = segmenter_instance.refine_mask(dummy_mask_array, operations=dilate_ops)
    assert np.sum(dilated > 0) > np.sum(dummy_mask_array > 0)

def test_refine_mask_invalid_op_type(segmenter_instance, dummy_mask_array):
    """Test that an invalid operation type raises ValueError."""
    invalid_ops = [{'type': 'blur', 'kernel_size': 3}]
    with pytest.raises(ValueError, match="Invalid operation type"): 
        segmenter_instance.refine_mask(dummy_mask_array, operations=invalid_ops)

def test_refine_mask_invalid_kernel(segmenter_instance, dummy_mask_array):
    """Test that invalid kernel sizes raise ValueError."""
    invalid_ops1 = [{'type': 'erode', 'kernel_size': 0}]
    invalid_ops2 = [{'type': 'erode', 'kernel_size': 4}] # Even number
    with pytest.raises(ValueError, match="Invalid kernel_size"): 
        segmenter_instance.refine_mask(dummy_mask_array, operations=invalid_ops1)
    with pytest.raises(ValueError, match="Invalid kernel_size"): 
        segmenter_instance.refine_mask(dummy_mask_array, operations=invalid_ops2)

def test_refine_mask_invalid_iterations(segmenter_instance, dummy_mask_array):
    """Test that invalid iteration counts raise ValueError."""
    invalid_ops = [{'type': 'erode', 'kernel_size': 3, 'iterations': 0}]
    with pytest.raises(ValueError, match="Invalid iterations"): 
        segmenter_instance.refine_mask(dummy_mask_array, operations=invalid_ops)

def test_refine_mask_invalid_mask_type(segmenter_instance):
    """Test passing non-numpy array raises TypeError."""
    with pytest.raises(TypeError):
        segmenter_instance.refine_mask([1, 2, 3]) # Pass a list

# --- Tests for Segmenter.evaluate_mask_quality --- 

def test_evaluate_mask_empty(segmenter_instance):
    """Test evaluation of a completely black mask."""
    empty_mask = np.zeros((TEST_HEIGHT, TEST_WIDTH), dtype=np.uint8)
    metrics = segmenter_instance.evaluate_mask_quality(empty_mask)
    assert metrics['coverage'] == 0
    assert metrics['complexity'] == 0
    assert metrics['contours'] == 0
    assert metrics['is_valid'] is False

def test_evaluate_mask_full(segmenter_instance):
    """Test evaluation of a completely white mask."""
    full_mask = np.ones((TEST_HEIGHT, TEST_WIDTH), dtype=np.uint8) * 255
    metrics = segmenter_instance.evaluate_mask_quality(full_mask)
    assert metrics['coverage'] == 100
    # Complexity might vary slightly based on contour finding of border
    assert metrics['contours'] >= 1 
    assert metrics['is_valid'] is False # Coverage > 95

def test_evaluate_mask_simple(segmenter_instance, dummy_mask_array):
    """Test evaluation of a simple mask."""
    metrics = segmenter_instance.evaluate_mask_quality(dummy_mask_array)
    assert 5 < metrics['coverage'] < 95
    assert metrics['complexity'] > 0
    assert metrics['contours'] == 1
    assert metrics['is_valid'] is True

def test_evaluate_mask_complex(segmenter_instance, complex_mask):
    """Test evaluation of a fragmented mask."""
    metrics = segmenter_instance.evaluate_mask_quality(complex_mask)
    assert 5 < metrics['coverage'] < 95
    assert metrics['complexity'] > 0
    assert metrics['contours'] > 1 # Expect multiple contours
    assert metrics['is_valid'] is True

def test_evaluate_mask_bool_dtype(segmenter_instance):
    """Test evaluation with a boolean mask."""
    bool_mask = np.zeros((TEST_HEIGHT, TEST_WIDTH), dtype=bool)
    bool_mask[10:20, 10:20] = True
    metrics = segmenter_instance.evaluate_mask_quality(bool_mask)
    assert 5 < metrics['coverage'] < 95
    assert metrics['contours'] == 1
    assert metrics['is_valid'] is True

# --- Tests for Segmenter.segment (using mocking) --- 

# Mock the rembg.remove function
@patch('rembg.remove')
def test_segment_path_input(mock_rembg_remove, segmenter_instance, tmp_path):
    """Test segmenting with a file path input."""
    # Arrange
    img_path = tmp_path / "test_segment.png"
    dummy_img_rgb = Image.new('RGB', (TEST_WIDTH, TEST_HEIGHT), color='red')
    dummy_img_rgba_output = Image.new('RGBA', (TEST_WIDTH, TEST_HEIGHT), color=(255, 0, 0, 255))
    dummy_img_rgb.save(img_path)
    mock_rembg_remove.return_value = dummy_img_rgba_output # Mock returns RGBA PIL
    
    # Act
    mask = segmenter_instance.segment(img_path, return_rgba=False)
    
    # Assert
    assert isinstance(mask, np.ndarray)
    assert mask.shape == (TEST_HEIGHT, TEST_WIDTH)
    assert mask.dtype == np.uint8
    assert np.all(mask == 255) # Check if alpha channel was extracted
    mock_rembg_remove.assert_called_once() # Check if rembg.remove was called
    # Can add more detailed checks on call arguments if needed
    call_args, call_kwargs = mock_rembg_remove.call_args
    assert isinstance(call_args[0], Image.Image)
    assert call_args[0].mode == 'RGB'
    assert call_kwargs['session'] == segmenter_instance.session

@patch('rembg.remove')
def test_segment_return_rgba(mock_rembg_remove, segmenter_instance, tmp_path):
    """Test the return_rgba=True flag."""
    img_path = tmp_path / "test_segment_rgba.png"
    dummy_img_rgb = Image.new('RGB', (TEST_WIDTH, TEST_HEIGHT), color='green')
    dummy_img_rgba_output = Image.new('RGBA', (TEST_WIDTH, TEST_HEIGHT), color=(0, 255, 0, 128))
    dummy_img_rgb.save(img_path)
    mock_rembg_remove.return_value = dummy_img_rgba_output
    
    rgba_out, mask_out = segmenter_instance.segment(img_path, return_rgba=True)
    
    assert isinstance(rgba_out, np.ndarray)
    assert rgba_out.shape == (TEST_HEIGHT, TEST_WIDTH, 4)
    assert rgba_out.dtype == np.uint8
    assert isinstance(mask_out, np.ndarray)
    assert mask_out.shape == (TEST_HEIGHT, TEST_WIDTH)
    assert mask_out.dtype == np.uint8
    assert np.all(mask_out == 128)
    mock_rembg_remove.assert_called_once()

@patch('rembg.remove')
def test_segment_numpy_input(mock_rembg_remove, segmenter_instance):
    """Test segmenting with a NumPy array input."""
    dummy_array_rgb = np.zeros((TEST_HEIGHT, TEST_WIDTH, 3), dtype=np.uint8)
    dummy_img_rgba_output = Image.new('RGBA', (TEST_WIDTH, TEST_HEIGHT), color=(0, 0, 0, 255))
    mock_rembg_remove.return_value = dummy_img_rgba_output

    mask = segmenter_instance.segment(dummy_array_rgb)
    assert isinstance(mask, np.ndarray)
    mock_rembg_remove.assert_called_once()
    call_args, _ = mock_rembg_remove.call_args
    assert isinstance(call_args[0], Image.Image) # Should be converted to PIL

@patch('rembg.remove')
def test_segment_pil_input(mock_rembg_remove, segmenter_instance):
    """Test segmenting with a PIL Image input."""
    dummy_img_rgb = Image.new('RGB', (TEST_WIDTH, TEST_HEIGHT), color='blue')
    dummy_img_rgba_output = Image.new('RGBA', (TEST_WIDTH, TEST_HEIGHT), color=(0, 0, 255, 255))
    mock_rembg_remove.return_value = dummy_img_rgba_output

    mask = segmenter_instance.segment(dummy_img_rgb)
    assert isinstance(mask, np.ndarray)
    mock_rembg_remove.assert_called_once()
    call_args, _ = mock_rembg_remove.call_args
    assert call_args[0] == dummy_img_rgb # Should pass PIL image directly

def test_segment_file_not_found(segmenter_instance):
    """Test FileNotFoundError is raised for non-existent path."""
    with pytest.raises(FileNotFoundError):
        segmenter_instance.segment("non/existent/image.png")

def test_segment_invalid_type(segmenter_instance):
    """Test TypeError is raised for invalid input type."""
    with pytest.raises(TypeError):
        segmenter_instance.segment([1, 2, 3]) # Pass a list 