# -*- coding: utf-8 -*-
"""
Tests for the main generation pipeline in src.pipeline.main_pipeline.
"""
import sys
import pytest
from pathlib import Path
import numpy as np
from PIL import Image
from unittest.mock import patch, MagicMock, call

# Ensure src is importable
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Class to be tested
from src.pipeline.main_pipeline import GenerationPipeline
# Modules/Classes to be mocked
import src.config
import src.utils.data_io 
import src.background.generators
import src.background.utils
import src.image.segmentation
import src.image.quality
import src.models.diffusion

# --- Constants & Fixtures ---
TEST_WIDTH = 64
TEST_HEIGHT = 48

@pytest.fixture
def mock_config():
    """Provides a mock default config dictionary."""
    # Simplified config for easier mocking control
    return {
        "paths": {},
        "segmentation": {"model": "u2net"},
        "diffusion": {"enabled": False, "sd_model_id": "dummy-sd", "controlnet_type": "seg"},
        "quality_assessment": {"min_resolution": (10, 10), "blur_threshold": 10, "contrast_threshold": 10, "bg_complexity_threshold": 0.1},
        "generation": {"add_shadow": False, "default_bg_type": "solid", "default_bg_color": (200, 200, 200)},
        "output": {"format": "png", "jpeg_quality": 90}
    }

@pytest.fixture
def dummy_pil_image() -> Image.Image:
    return Image.new('RGB', (TEST_WIDTH, TEST_HEIGHT), color=(100, 150, 200))

@pytest.fixture
def dummy_mask_array() -> np.ndarray:
    mask = np.zeros((TEST_HEIGHT, TEST_WIDTH), dtype=np.uint8)
    mask[5:-5, 5:-5] = 255
    return mask

@pytest.fixture
def dummy_background_array() -> np.ndarray:
    return np.ones((TEST_HEIGHT, TEST_WIDTH, 3), dtype=np.uint8) * 128

# --- Mock Setup using pytest fixture and patch --- 

@pytest.fixture(autouse=True)
def mock_pipeline_dependencies(mocker, mock_config, dummy_pil_image, dummy_mask_array, dummy_background_array):
    """Mocks all external dependencies used by GenerationPipeline for most tests."""
    # Mock config loading
    mocker.patch('src.config.load_config', return_value=mock_config) 

    # Mock IO functions
    mocker.patch('src.utils.data_io.load_image', return_value=dummy_pil_image)
    mocker.patch('src.utils.data_io.save_image', return_value=True) # Assume save succeeds
    mocker.patch('src.background.utils.load_background_image', return_value=dummy_background_array)
    
    # Mock Generator functions
    mocker.patch('src.background.generators.generate_solid_background', return_value=Image.new('RGB', (TEST_WIDTH, TEST_HEIGHT), (1, 2, 3)))
    mocker.patch('src.background.generators.generate_gradient_background', return_value=Image.new('RGB', (TEST_WIDTH, TEST_HEIGHT), (4, 5, 6)))
    
    # Mock Background utils
    mocker.patch('src.background.utils.combine_foreground_background', return_value=np.array(dummy_pil_image))
    mocker.patch('src.background.utils.add_simple_drop_shadow', return_value=np.array(dummy_pil_image))

    # Mock Classes (Patch the class itself)
    mock_segmenter_instance = MagicMock()
    mock_segmenter_instance.segment.return_value = dummy_mask_array
    mock_segmenter_instance.refine_mask.return_value = dummy_mask_array
    mocker.patch('src.image.segmentation.Segmenter', return_value=mock_segmenter_instance)

    mock_assessor_instance = MagicMock()
    mock_assessor_instance.assess_quality.return_value = {'is_processable': True, 'message': 'Mock quality OK'}
    mocker.patch('src.image.quality.ImageAssessor', return_value=mock_assessor_instance)

    mock_diffusion_instance = MagicMock()
    mock_diffusion_instance.generate.return_value = Image.new('RGB', (TEST_WIDTH, TEST_HEIGHT), (10,20,30))
    mocker.patch('src.models.diffusion.DiffusionGenerator', return_value=mock_diffusion_instance)
    
    # Return the mocks if needed in tests, though mocker provides access too
    return {
        "load_config": src.config.load_config,
        "load_image": src.utils.data_io.load_image,
        "save_image": src.utils.data_io.save_image,
        "load_background_image": src.background.utils.load_background_image,
        "generate_solid": src.background.generators.generate_solid_background,
        "generate_gradient": src.background.generators.generate_gradient_background,
        "combine": src.background.utils.combine_foreground_background,
        "add_shadow": src.background.utils.add_simple_drop_shadow,
        "Segmenter": src.image.segmentation.Segmenter,
        "ImageAssessor": src.image.quality.ImageAssessor,
        "DiffusionGenerator": src.models.diffusion.DiffusionGenerator,
        "mock_segmenter_instance": mock_segmenter_instance,
        "mock_assessor_instance": mock_assessor_instance,
        "mock_diffusion_instance": mock_diffusion_instance
    }

# --- Tests for Pipeline Initialization --- 

def test_pipeline_init_defaults(mock_pipeline_dependencies):
    """Test pipeline initializes with default config and mocks."""
    pipeline = GenerationPipeline() 
    assert pipeline is not None
    assert pipeline.config == mock_pipeline_dependencies['load_config'].return_value
    # Check if component classes were instantiated (mocked)
    mock_pipeline_dependencies['Segmenter'].assert_called_once()
    mock_pipeline_dependencies['ImageAssessor'].assert_called_once()
    # Diffusion should not be initialized by default in mock_config
    mock_pipeline_dependencies['DiffusionGenerator'].assert_not_called()
    assert pipeline.diffusion_generator is None

def test_pipeline_init_diffusion_enabled(mock_pipeline_dependencies):
    """Test pipeline initializes DiffusionGenerator when enabled."""
    # Override the default mock config for this test
    mock_config = mock_pipeline_dependencies['load_config'].return_value
    mock_config['diffusion']['enabled'] = True # Enable diffusion in mock config
    
    pipeline = GenerationPipeline(diffusion_enabled=True) # Also override via arg
    assert pipeline.diffusion_generator is not None
    mock_pipeline_dependencies['DiffusionGenerator'].assert_called_once()
    # Check args passed to DiffusionGenerator based on mock_config
    call_args, call_kwargs = mock_pipeline_dependencies['DiffusionGenerator'].call_args
    assert call_kwargs['sd_model_id'] == mock_config['diffusion']['sd_model_id']

def test_pipeline_init_diffusion_override(mock_pipeline_dependencies):
    """Test overriding diffusion config during initialization."""
    override = {'controlnet_type': 'canny', 'device': 'cpu'}
    pipeline = GenerationPipeline(diffusion_enabled=True, diffusion_cfg_overrides=override)
    assert pipeline.diffusion_generator is not None
    mock_pipeline_dependencies['DiffusionGenerator'].assert_called_once()
    call_args, call_kwargs = mock_pipeline_dependencies['DiffusionGenerator'].call_args
    assert call_kwargs['controlnet_type'] == 'canny' # Overridden
    assert call_kwargs['device'] == 'cpu' # Overridden
    assert call_kwargs['sd_model_id'] == mock_pipeline_dependencies['load_config'].return_value['diffusion']['sd_model_id'] # Default used

# --- Tests for process_image Orchestration ---

def test_process_image_success_solid_bg(mock_pipeline_dependencies, tmp_path):
    """Test the success path of process_image with solid background spec."""
    pipeline = GenerationPipeline()
    input_path = "dummy_input.png"
    output_path = tmp_path / "output.png"
    bg_spec = (10, 20, 30) # Solid color tuple
    
    success = pipeline.process_image(input_path, output_path, bg_spec)
    
    assert success is True
    # Check mocks were called
    mock_pipeline_dependencies['load_image'].assert_called_once_with(input_path, mode='RGB')
    mock_pipeline_dependencies['mock_assessor_instance'].assess_quality.assert_called_once()
    mock_pipeline_dependencies['mock_segmenter_instance'].segment.assert_called_once()
    mock_pipeline_dependencies['mock_segmenter_instance'].refine_mask.assert_called_once()
    mock_pipeline_dependencies['generate_solid'].assert_called_once_with(width=TEST_WIDTH, height=TEST_HEIGHT, color=bg_spec)
    mock_pipeline_dependencies['load_background_image'].assert_not_called()
    mock_pipeline_dependencies['mock_diffusion_instance'].generate.assert_not_called()
    mock_pipeline_dependencies['combine'].assert_called_once()
    # Shadow is False by default in mock config
    mock_pipeline_dependencies['add_shadow'].assert_not_called()
    mock_pipeline_dependencies['save_image'].assert_called_once()
    # Check save path argument
    call_args, call_kwargs = mock_pipeline_dependencies['save_image'].call_args
    assert call_kwargs['save_path'] == output_path.with_suffix('.png') # Check format applied

def test_process_image_success_gradient_bg(mock_pipeline_dependencies, tmp_path):
    """Test success path with gradient dict spec."""
    pipeline = GenerationPipeline()
    input_path = "dummy_input.png"
    output_path = tmp_path / "output.jpg"
    bg_spec = {'type': 'gradient', 'colors': [(1,1,1), (2,2,2)], 'direction': 'radial'}
    
    success = pipeline.process_image(input_path, output_path, bg_spec)
    
    assert success is True
    mock_pipeline_dependencies['generate_gradient'].assert_called_once_with(width=TEST_WIDTH, height=TEST_HEIGHT, colors=bg_spec['colors'], direction=bg_spec['direction'])
    mock_pipeline_dependencies['save_image'].assert_called_once()
    call_args, call_kwargs = mock_pipeline_dependencies['save_image'].call_args
    assert call_kwargs['save_path'] == output_path.with_suffix('.jpg') # Check format applied
    assert call_kwargs['quality'] == 90 # Default JPEG quality from mock config

def test_process_image_success_file_bg(mock_pipeline_dependencies, tmp_path):
    """Test success path with background file path spec."""
    pipeline = GenerationPipeline()
    input_path = "dummy_input.png"
    output_path = tmp_path / "output.png"
    bg_spec = tmp_path / "my_background.jpg"
    
    success = pipeline.process_image(input_path, output_path, bg_spec)
    
    assert success is True
    mock_pipeline_dependencies['load_background_image'].assert_called_once_with(bg_spec, (TEST_WIDTH, TEST_HEIGHT))
    mock_pipeline_dependencies['generate_solid'].assert_not_called()
    mock_pipeline_dependencies['generate_gradient'].assert_not_called()
    mock_pipeline_dependencies['save_image'].assert_called_once()

@patch('src.models.diffusion.DiffusionGenerator') # Need fresh mock inside test
def test_process_image_success_diffusion_bg(MockDiffGen, mock_pipeline_dependencies, tmp_path):
    """Test success path with diffusion background spec."""
    # Re-configure mocks for this specific test to enable diffusion
    mock_config = mock_pipeline_dependencies['load_config'].return_value
    mock_config['diffusion']['enabled'] = True 
    # Setup instance mock for the generator created *inside* the pipeline init
    mock_diff_instance = MagicMock()
    mock_diff_instance.generate.return_value = Image.new('RGB', (TEST_WIDTH, TEST_HEIGHT))
    MockDiffGen.return_value = mock_diff_instance

    pipeline = GenerationPipeline(diffusion_enabled=True) # Enable diffusion
    input_path = "dummy_input.png"
    output_path = tmp_path / "output.png"
    bg_spec = {'type': 'diffusion'} # Prompt comes from arg
    prompt = "test diffusion prompt"
    
    success = pipeline.process_image(input_path, output_path, bg_spec, prompt=prompt)
    
    assert success is True
    MockDiffGen.assert_called_once() # Check generator was initialized
    mock_diff_instance.generate.assert_called_once() # Check generate was called
    # Check prompt was passed correctly
    call_args, call_kwargs = mock_diff_instance.generate.call_args
    assert call_kwargs['prompt'] == prompt 
    mock_pipeline_dependencies['generate_solid'].assert_not_called()
    mock_pipeline_dependencies['generate_gradient'].assert_not_called()
    mock_pipeline_dependencies['load_background_image'].assert_not_called()
    mock_pipeline_dependencies['save_image'].assert_called_once()

def test_process_image_with_shadow(mock_pipeline_dependencies, tmp_path):
    """Test shadow is added when config enables it."""
    mock_config = mock_pipeline_dependencies['load_config'].return_value
    mock_config['generation']['add_shadow'] = True # Enable shadow in mock config
    
    pipeline = GenerationPipeline()
    input_path = "dummy_input.png"
    output_path = tmp_path / "output.png"
    bg_spec = (10, 20, 30)
    
    success = pipeline.process_image(input_path, output_path, bg_spec)
    
    assert success is True
    mock_pipeline_dependencies['add_shadow'].assert_called_once()
    mock_pipeline_dependencies['save_image'].assert_called_once()

def test_process_image_load_fail(mock_pipeline_dependencies, tmp_path):
    """Test pipeline returns False if initial image load fails."""
    # Override mock for load_image to simulate failure
    mock_pipeline_dependencies['load_image'].return_value = None 
    
    pipeline = GenerationPipeline()
    input_path = "fail_load.png"
    output_path = tmp_path / "output.png"
    bg_spec = (10, 20, 30)
    
    success = pipeline.process_image(input_path, output_path, bg_spec)
    assert success is False
    # Check that subsequent steps were not called
    mock_pipeline_dependencies['mock_assessor_instance'].assess_quality.assert_not_called()
    mock_pipeline_dependencies['mock_segmenter_instance'].segment.assert_not_called()
    mock_pipeline_dependencies['save_image'].assert_not_called()

def test_process_image_segment_fail(mock_pipeline_dependencies, tmp_path):
    """Test pipeline returns False if segmentation fails."""
    # Override mock for segmenter to simulate failure
    mock_pipeline_dependencies['mock_segmenter_instance'].segment.side_effect = Exception("Segmentation Error")
    
    pipeline = GenerationPipeline()
    input_path = "segment_fail.png"
    output_path = tmp_path / "output.png"
    bg_spec = (10, 20, 30)
    
    success = pipeline.process_image(input_path, output_path, bg_spec)
    assert success is False
    mock_pipeline_dependencies['load_image'].assert_called_once()
    mock_pipeline_dependencies['mock_assessor_instance'].assess_quality.assert_called_once()
    mock_pipeline_dependencies['mock_segmenter_instance'].segment.assert_called_once()
    # Check that steps after segmentation were not called
    mock_pipeline_dependencies['mock_segmenter_instance'].refine_mask.assert_not_called()
    mock_pipeline_dependencies['generate_solid'].assert_not_called()
    mock_pipeline_dependencies['save_image'].assert_not_called()

def test_process_image_save_fail(mock_pipeline_dependencies, tmp_path):
    """Test pipeline returns False if saving the final image fails."""
    # Override mock for save_image to simulate failure
    mock_pipeline_dependencies['save_image'].return_value = False
    
    pipeline = GenerationPipeline()
    input_path = "save_fail.png"
    output_path = tmp_path / "output.png"
    bg_spec = (10, 20, 30)
    
    success = pipeline.process_image(input_path, output_path, bg_spec)
    assert success is False
    mock_pipeline_dependencies['save_image'].assert_called_once() # Should still be called

# TODO: Add more tests for edge cases, different config combinations, etc. 