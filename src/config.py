# -*- coding: utf-8 -*-
"""
Central configuration settings for the background generation pipeline.

Consider using a more robust configuration library (like OmegaConf or Dynaconf)
for more complex scenarios, but simple dict/constants are fine for now.
"""

from typing import Dict, Any, Tuple, List, Optional
import os

# Get the project root directory (assuming config.py is in src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Consolidated Default Configuration Dictionary ---
# This is the single source of truth for default settings.
# Overridden by pipeline init args or CLI flags where applicable.
DEFAULT_CONFIG: Dict[str, Any] = {
    # --- Input Image Processing ---
    "min_resolution": (512, 512),      # Minimum width, height
    "target_resolution": (1024, 1024), # Target resolution for processing if scaling (Currently unused in pipeline)
    "check_resolution": True,          # Flags for ImageAssessor (Currently unused in pipeline)
    "check_blur": True,
    "blur_threshold": 100,             # Laplacian variance threshold
    "check_contrast": True,
    "contrast_threshold": 0.1,         # RMS contrast

    # --- Segmentation ---
    "segmenter_model": "u2net",         # Default rembg model (e.g., u2net, u2netp, silueta, isnet-general-use)
    "segmentation_device": "cpu",       # Device for segmentation model ("cpu" or "cuda")
    "refine_mask": True,                # Apply morphological refinement?
    # Mask refinement parameters (applied if refine_mask is True)
    "mask_opening_kernel_size": 5,      # Kernel size for morphological opening (removes noise, must be positive odd)
    "mask_opening_iterations": 1,       # Iterations for opening
    "mask_dilation_kernel_size": 5,     # Kernel size for morphological dilation (expands mask, must be positive odd)
    "mask_dilation_iterations": 1,      # Iterations for dilation

    # --- Compositing ---
    "add_shadow": True,                 # Add drop shadow?
    "edge_feathering_amount": 2.0,      # Sigma for Gaussian blur on mask edge (0 or less disables)

    # Shadow parameters (applied if add_shadow is True)
    "shadow_offset_x": 5,               # Shadow horizontal offset in pixels
    "shadow_offset_y": 5,               # Shadow vertical offset in pixels
    "shadow_blur_sigma": 5.0,           # Gaussian blur sigma for shadow softness (0 or less disables blur)
    "shadow_opacity": 0.5,              # Opacity of the shadow (0.0 to 1.0)
    "shadow_color": (0, 0, 0),          # Shadow color (RGB)

    # --- Background Generation ---
    "diffusion_enabled": False,         # Enable diffusion background generation?
    # Default Diffusion Settings (can be overridden by diffusion_cfg)
    "diffusion_model_id": "runwayml/stable-diffusion-v1-5",
    "diffusion_controlnet_model_id": "lllyasviel/control_v11p_sd15_inpaint", # Or other ControlNets
    "diffusion_device": "cpu",          # "cuda" if GPU available/configured
    "diffusion_num_inference_steps": 25,
    "diffusion_guidance_scale": 7.5,
    "diffusion_controlnet_scale": 1.0,
    "diffusion_enable_cpu_offload": True, # If using GPU, offload when possible
    "diffusion_scheduler": "DDIM",      # Scheduler type (e.g., DDIM, UniPC)

    # --- Output ---
    "default_output_dir": os.path.join(PROJECT_ROOT, "results", "pipeline_outputs"),
    "save_intermediate_masks": False,   # Save raw/refined masks for debugging?
    "intermediate_mask_dir": os.path.join(PROJECT_ROOT, "results", "intermediate_masks"),
    "output_format": "png",             # Default save format (png, jpg, webp)
    "output_jpeg_quality": 90,          # Quality for JPEG/WebP saving

    # --- Legacy/Unused (Placeholder, can be removed if confirmed unused) ---
    # "paths": { ... }, # Replaced by direct path joins above or handled elsewhere
    # "quality_assessment": { ... } # Keys moved to top level
    # "generation": { ... } # Keys moved to top level or specific sections
}

# --- Function to load config --- 
def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Loads configuration.
    Currently returns defaults, TODO: Implement loading from file (YAML/JSON).
    """
    if config_path:
        print(f"Warning: Loading config from file ({config_path}) not yet implemented. Using defaults.")
        # TODO: Implement loading logic here (e.g., using PyYAML or json)
        # Example:
        # try:
        #     with open(config_path, 'r') as f:
        #         loaded_config = yaml.safe_load(f) # if using YAML
        #     # Deep merge might be needed depending on structure
        #     merged_config = DEFAULT_CONFIG.copy()
        #     # Simple top-level merge example:
        #     for key, value in loaded_config.items():
        #         if key in merged_config and isinstance(merged_config[key], dict) and isinstance(value, dict):
        #              merged_config[key].update(value)
        #         else:
        #              merged_config[key] = value
        #     return merged_config
        # except Exception as e:
        #     print(f"Error loading config file {config_path}: {e}. Falling back to defaults.")

    # Return a copy to prevent modification of the global default
    return DEFAULT_CONFIG.copy()

def get_config_value(key: str, default_value: Optional[Any] = None) -> Any:
    """Helper function to get a specific config value from the loaded config."""
    # Always load the config to ensure potential file loading is handled
    config = load_config()
    # Use get for safe access, returning the provided default if key is missing
    return config.get(key, default_value) 