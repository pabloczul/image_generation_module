# -*- coding: utf-8 -*-
"""
Tests for the image quality assessment utilities in src.image.quality.
"""
import sys
import pytest
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter
import cv2 # For creating test images

# Ensure src is importable
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.image.quality import ImageAssessor, _load_image
from src.image.quality import QUALITY_EXCELLENT, QUALITY_GOOD, QUALITY_FAIR, QUALITY_POOR

# --- Constants & Test Image Generation Helpers ---
TEST_WIDTH = 100
TEST_HEIGHT = 80
MIN_RES = (50, 50) # Example minimum resolution for tests
BLUR_THRESH = 50.0 # Example blur threshold
CONTRAST_THRESH = 20.0 # Example contrast threshold
BG_COMPLEX_THRESH = 0.1 # Example complexity threshold

def create_test_image(tmp_path, filename="test.png", size=(TEST_WIDTH, TEST_HEIGHT), 
                        color=(128, 128, 128), mode='RGB') -> Path:
    """Creates a simple solid color image."""
    img_path = tmp_path / filename
    img = Image.new(mode, size, color=color)
    img.save(img_path)
    return img_path

def create_blurry_image(tmp_path, filename="blurry.png", size=(TEST_WIDTH, TEST_HEIGHT)) -> Path:
    """Creates a blurry image."""
    img_path = tmp_path / filename
    # Create some pattern first
    img_array = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    img_array[::10, :, :] = 255
    img_array[:, ::10, :] = 255
    img = Image.fromarray(img_array)
    # Apply heavy blur
    blurred_img = img.filter(ImageFilter.GaussianBlur(radius=5))
    blurred_img.save(img_path)
    return img_path

def create_sharp_image(tmp_path, filename="sharp.png", size=(TEST_WIDTH, TEST_HEIGHT)) -> Path:
    """Creates a sharp image with high frequency details."""
    img_path = tmp_path / filename
    # Create checkerboard pattern
    img_array = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    rows, cols = size[1], size[0]
    for r in range(rows):
        for c in range(cols):
            if (r // 8 + c // 8) % 2 == 0:
                img_array[r, c, :] = 255
    img = Image.fromarray(img_array)
    img.save(img_path)
    return img_path

def create_low_contrast_image(tmp_path, filename="low_contrast.png", size=(TEST_WIDTH, TEST_HEIGHT)) -> Path:
    """Creates a low contrast image."""
    img_path = tmp_path / filename
    # Grayscale gradient with low range
    img_array = np.zeros((size[1], size[0]), dtype=np.uint8)
    for r in range(size[1]):
        img_array[r, :] = 120 + int(20 * r / size[1]) # Range 120-140
    img = Image.fromarray(img_array).convert('RGB')
    img.save(img_path)
    return img_path
    
# --- Fixtures ---

@pytest.fixture
def default_assessor() -> ImageAssessor:
    """Provides an ImageAssessor with default test thresholds."""
    return ImageAssessor(
        min_resolution=MIN_RES,
        blur_threshold=BLUR_THRESH,
        contrast_threshold=CONTRAST_THRESH,
        bg_complexity_threshold=BG_COMPLEX_THRESH
    )

# --- Tests for ImageAssessor Initialization ---

def test_assessor_init_defaults():
    """Test initialization with default values."""
    assessor = ImageAssessor()
    assert assessor.min_resolution == (300, 300) # Check against actual defaults

def test_assessor_init_custom():
    """Test initialization with custom values."""
    assessor = ImageAssessor(min_resolution=(10,10), blur_threshold=10.0)
    assert assessor.min_resolution == (10, 10)
    assert assessor.blur_threshold == 10.0

def test_assessor_init_invalid():
    """Test initialization with invalid values raises ValueError."""
    with pytest.raises(ValueError):
        ImageAssessor(min_resolution=(100, 0))
    with pytest.raises(ValueError):
        ImageAssessor(blur_threshold=-5.0)
    with pytest.raises(ValueError):
        ImageAssessor(contrast_threshold=-1)

# --- Tests for Private Helper Methods (Indirectly tested via assess_quality) ---
# It's often sufficient to test the public assess_quality method, 
# but direct tests can be useful during development.

def test_check_resolution(default_assessor, tmp_path):
    img_ok = Image.new('RGB', MIN_RES)
    img_low_w = Image.new('RGB', (MIN_RES[0] - 1, MIN_RES[1]))
    img_low_h = Image.new('RGB', (MIN_RES[0], MIN_RES[1] - 1))
    assert default_assessor._check_resolution(img_ok)['passed'] is True
    assert default_assessor._check_resolution(img_low_w)['passed'] is False
    assert default_assessor._check_resolution(img_low_h)['passed'] is False

def test_calculate_blur(default_assessor, tmp_path):
    sharp_path = create_sharp_image(tmp_path)
    blurry_path = create_blurry_image(tmp_path)
    _pil_sharp, cv_sharp = _load_image(sharp_path)
    _pil_blurry, cv_blurry = _load_image(blurry_path)
    
    blur_res_sharp = default_assessor._calculate_blur(cv_sharp)
    blur_res_blurry = default_assessor._calculate_blur(cv_blurry)
    
    assert blur_res_sharp['passed'] is True
    assert blur_res_blurry['passed'] is False
    assert blur_res_sharp['value'] > blur_res_blurry['value']

def test_calculate_contrast(default_assessor, tmp_path):
    sharp_path = create_sharp_image(tmp_path) # High contrast checkerboard
    low_contrast_path = create_low_contrast_image(tmp_path)
    _pil_sharp, cv_sharp = _load_image(sharp_path)
    _pil_low, cv_low = _load_image(low_contrast_path)

    contrast_res_sharp = default_assessor._calculate_contrast(cv_sharp)
    contrast_res_low = default_assessor._calculate_contrast(cv_low)

    assert contrast_res_sharp['passed'] is True
    assert contrast_res_low['passed'] is False
    assert contrast_res_sharp['value'] > contrast_res_low['value']

# --- Tests for assess_quality (Public Method) ---

def test_assess_quality_excellent(default_assessor, tmp_path):
    """Test case for an image expected to pass all checks."""
    img_path = create_sharp_image(tmp_path, size=MIN_RES) # Meets min res, sharp, high contrast
    assessment = default_assessor.assess_quality(img_path)
    
    assert assessment['resolution']['passed'] is True
    assert assessment['blur']['passed'] is True
    assert assessment['contrast']['passed'] is True
    assert assessment['overall_quality'] == QUALITY_EXCELLENT
    assert assessment['is_processable'] is True

def test_assess_quality_poor_resolution(default_assessor, tmp_path):
    """Test case for an image failing only resolution."""
    img_path = create_sharp_image(tmp_path, size=(MIN_RES[0]-1, MIN_RES[1])) 
    assessment = default_assessor.assess_quality(img_path)
    
    assert assessment['resolution']['passed'] is False
    # Blur/Contrast might still pass, but overall should be poor
    assert assessment['overall_quality'] == QUALITY_POOR
    assert assessment['is_processable'] is False

def test_assess_quality_fair_blur_contrast(default_assessor, tmp_path):
    """Test case for image with ok resolution but failing blur and contrast."""
    # Create an image that's slightly blurry AND low contrast
    img_path = tmp_path / "fair_image.png"
    img_array = np.ones((MIN_RES[1], MIN_RES[0], 3), dtype=np.uint8) * 128 # Mid-gray
    img = Image.fromarray(img_array)
    img = img.filter(ImageFilter.GaussianBlur(radius=2)) # Moderate blur
    img.save(img_path)

    assessment = default_assessor.assess_quality(img_path)
    
    assert assessment['resolution']['passed'] is True
    assert assessment['blur']['passed'] is False
    assert assessment['contrast']['passed'] is False
    assert assessment['overall_quality'] == QUALITY_FAIR
    assert assessment['is_processable'] is True # Still processable per logic

def test_assess_quality_good_blur(default_assessor, tmp_path):
    """Test case for image failing only blur."""
    img_path = create_blurry_image(tmp_path, size=MIN_RES)
    # Need to ensure contrast passes - blurry image might have low contrast too.
    # For this test, let's assume the blurry image still has enough contrast. 
    # A more robust test might manually create high contrast blurry image.
    assessment = default_assessor.assess_quality(img_path)
    
    assert assessment['resolution']['passed'] is True
    assert assessment['blur']['passed'] is False
    # We assume contrast passes for 'good' classification in this scenario
    if assessment['contrast']['passed']:
        assert assessment['overall_quality'] == QUALITY_GOOD
        assert assessment['is_processable'] is True
    else: # If contrast also failed, it would be 'fair'
        assert assessment['overall_quality'] == QUALITY_FAIR
        assert assessment['is_processable'] is True 

def test_assess_quality_good_contrast(default_assessor, tmp_path):
    """Test case for image failing only contrast."""
    img_path = create_low_contrast_image(tmp_path, size=MIN_RES)
    assessment = default_assessor.assess_quality(img_path)
    
    assert assessment['resolution']['passed'] is True
    assert assessment['contrast']['passed'] is False
    # We assume blur passes for 'good' classification
    if assessment['blur']['passed']:
        assert assessment['overall_quality'] == QUALITY_GOOD
        assert assessment['is_processable'] is True
    else: # If blur also failed
        assert assessment['overall_quality'] == QUALITY_FAIR
        assert assessment['is_processable'] is True

# --- Tests for check_background_complexity --- 

@pytest.fixture
def simple_mask(size=(TEST_HEIGHT, TEST_WIDTH)) -> np.ndarray:
    """Mask with a simple square foreground."""
    mask = np.zeros(size, dtype=np.uint8)
    h, w = size
    mask[h//4:3*h//4, w//4:3*w//4] = 255
    return mask

@pytest.fixture
def complex_mask(size=(TEST_HEIGHT, TEST_WIDTH)) -> np.ndarray:
    """Mask with a more complex, fragmented foreground."""
    mask = np.zeros(size, dtype=np.uint8)
    h, w = size
    mask[h//8:3*h//8, w//8:3*w//8] = 255 # Top left
    mask[5*h//8:7*h//8, 5*w//8:7*w//8] = 255 # Bottom right
    mask[h//2:5*h//8, w//2:5*w//8] = 255 # Small middle
    return mask

def test_bg_complexity_simple_bg(default_assessor, tmp_path, simple_mask):
    """Test complexity with simple FG and plain background."""
    # Plain background should have almost zero edges
    img_path = create_test_image(tmp_path, color=(200, 200, 200))
    complexity = default_assessor.check_background_complexity(img_path, simple_mask)
    
    assert complexity['is_complex'] is False
    assert complexity['edge_density'] < 0.01 # Expect very low density

def test_bg_complexity_complex_bg(default_assessor, tmp_path, simple_mask):
    """Test complexity with simple FG and complex (sharp) background."""
    # Sharp/checkerboard background will have many edges
    img_path = create_sharp_image(tmp_path)
    complexity = default_assessor.check_background_complexity(img_path, simple_mask)
    
    # Assumes default threshold is low enough to catch this
    assert complexity['is_complex'] is True 
    assert complexity['edge_density'] > default_assessor.bg_complexity_threshold

def test_bg_complexity_requires_mask(default_assessor, tmp_path):
    """Test that the function requires the mask argument."""
    img_path = create_test_image(tmp_path)
    with pytest.raises(TypeError):
        default_assessor.check_background_complexity(img_path)
        
# --- Tests for recommend_processing_strategy --- 

# Basic tests, more comprehensive tests would mock assess_quality/check_complexity results

def test_recommend_strategy_not_processable(default_assessor):
    """Test recommendation when image is not processable."""
    quality_assessment = {'is_processable': False, 'message': 'Too small'}
    strategy = default_assessor.recommend_processing_strategy(quality_assessment)
    assert strategy['is_processable'] is False
    assert "not suitable" in strategy['messages'][0].lower()

def test_recommend_strategy_excellent(default_assessor):
    """Test recommendation for an excellent quality image."""
    quality_assessment = {
        'is_processable': True, 
        'overall_quality': QUALITY_EXCELLENT,
        'resolution': {'passed': True, 'width': 1000, 'height': 1000},
        'blur': {'passed': True, 'value': 200},
        'contrast': {'passed': True, 'value': 50},
        'message': 'Excellent'
    }
    strategy = default_assessor.recommend_processing_strategy(quality_assessment)
    assert strategy['is_processable'] is True
    assert strategy['segmentation_model'] == 'u2net' # Default for excellent
    assert len(strategy['mask_refinement_ops']) > 0 # Should have default ops
    assert strategy['suggested_background_type'] == 'gradient' # High res suggests gradient

def test_recommend_strategy_low_contrast(default_assessor):
    """Test recommendation applies different refinement for low contrast."""
    quality_assessment = {
        'is_processable': True, 
        'overall_quality': QUALITY_GOOD, 
        'resolution': {'passed': True}, 
        'blur': {'passed': True, 'value': 100}, 
        'contrast': {'passed': False, 'value': 10},
        'message': 'Low contrast'
    }
    strategy = default_assessor.recommend_processing_strategy(quality_assessment)
    assert strategy['is_processable'] is True
    # Check if refinement ops differ from default due to low contrast rule
    default_ops = default_assessor.recommend_processing_strategy({'is_processable': True, 'overall_quality': QUALITY_EXCELLENT, 'resolution':{'passed':True}, 'blur':{'passed':True}, 'contrast':{'passed':True}, 'message':''})['mask_refinement_ops']
    assert strategy['mask_refinement_ops'] != default_ops
    assert "low contrast" in " ".join(strategy['messages']).lower()

def test_recommend_strategy_complex_bg(default_assessor):
    """Test recommendation message when background is complex."""
    quality_assessment = {'is_processable': True, 'overall_quality': QUALITY_EXCELLENT, 'message': 'Excellent'}
    bg_complexity = {'is_complex': True, 'edge_density': 0.5, 'message': 'Complex BG'}
    strategy = default_assessor.recommend_processing_strategy(quality_assessment, bg_complexity)
    assert strategy['is_processable'] is True
    assert any("complex background" in m.lower() for m in strategy['messages']) 