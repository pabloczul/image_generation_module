"""
Script to run segmentation experiments based on configurations.

Reads parameters, configures the segmentation pipeline (segmenter + filters),
processes a dataset, and saves results (masks, visualizations) to a structured
directory named after the configuration.
"""

import argparse
import logging
import sys
import os
import json
from pathlib import Path
from typing import Dict, Any
import cv2
import numpy as np
from PIL import Image

# --- Setup Project Root and Imports ---
# Get the absolute path of the directory containing this script
script_dir = Path(__file__).resolve().parent 
# Get the absolute path of the parent directory (assumed project root)
project_root = script_dir.parent 

# Add both the script's directory and the project root directory to sys.path
# This ensures modules in 'scripts' and sibling directories like 'src' can be found
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir)) # Insert at beginning for potential precedence
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root)) # Insert project root as well

# Now try the imports - they should work if 'src' is directly under project_root
try:
    # These imports now rely on project_root being in sys.path
    from src.image.segmentation import Segmenter
    from src.image import filtering # Import the whole module
    from src.config import load_config, DEFAULT_CONFIG
    from src.utils.data_io import save_image, load_image
except ImportError as e:
    print(f"Error importing module components: {e}\n")
    print("Attempted to add script and project directories to sys.path.")
    print(f"Script Directory: {script_dir}")
    print(f"Project Root: {project_root}")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)
# -------------------------------------

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    stream=sys.stdout
)
# ---------------------

def create_run_name(config: Dict[str, Any]) -> str:
    """Creates a descriptive directory name from config parameters."""
    # Customize this to include the most relevant parameters for your experiments
    name_parts = [
        f"model-{config.get('segmenter_model', 'na')}",
        f"refine-{config.get('refine_mask', False)}",
    ]
    # Only add morph params if refine is True, otherwise name gets too long/redundant
    if config.get('refine_mask', False):
        name_parts.extend([
            f"ok-{config.get('mask_opening_kernel_size', 'na')}",
            f"oi-{config.get('mask_opening_iterations', 'na')}",
            f"ck-{config.get('mask_closing_kernel_size', 'na')}",
            f"ci-{config.get('mask_closing_iterations', 'na')}"
        ])
        
    name_parts.append(f"feather-{config.get('edge_feathering_amount', 'na')}")
    
    # Add filter statuses (keep concise)
    if config.get('apply_contrast_filter', False):
         name_parts.append(f"contrast-{config.get('contrast_filter_threshold', 'na')}")
    if config.get('apply_clutter_filter', False):
         name_parts.append(f"clutter-{config.get('clutter_min_primary_iou', 'na')}")
    if config.get('apply_contour_filter', False):
         name_parts.append(f"contour-{config.get('contour_min_solidity', 'na')}")

    # Limit overall length if necessary (optional)
    full_name = "_".join(name_parts)
    # max_len = 100 
    # if len(full_name) > max_len:
    #     import hashlib
    #     hash_part = hashlib.md5(full_name.encode()).hexdigest()[:8]
    #     return full_name[:max_len - 9] + '-' + hash_part
    return full_name

def create_visualization(original_img: np.ndarray, raw_mask: np.ndarray, final_mask: np.ndarray) -> np.ndarray:
    """Creates a side-by-side visualization: Original | Raw Mask | Final Mask | Overlay."""
    h, w = original_img.shape[:2]
    vis_h = h
    vis_w = w * 4 # Four images side-by-side

    visualization = np.zeros((vis_h, vis_w, 3), dtype=np.uint8)

    # Original Image (ensure 3 channels)
    if original_img.ndim == 2:
        original_3c = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
    else:
        original_3c = original_img
    visualization[0:h, 0:w] = original_3c

    # Raw Mask (ensure 3 channels, white on black)
    raw_mask_3c = cv2.cvtColor(raw_mask, cv2.COLOR_GRAY2BGR)
    visualization[0:h, w:w*2] = raw_mask_3c

    # Final Mask (ensure 3 channels, white on black)
    final_mask_3c = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR)
    visualization[0:h, w*2:w*3] = final_mask_3c

    # Overlay (Original + Final Mask blended)
    overlay = original_3c.copy()
    overlay[final_mask == 255] = [0, 255, 0] # Example: Green overlay
    # Add alpha blending for better visualization if desired
    # alpha = 0.4
    # cv2.addWeighted(overlay, alpha, original_3c, 1 - alpha, 0, overlay)
    visualization[0:h, w*3:w*4] = overlay

    return visualization

def run_single_experiment(config: Dict[str, Any], input_dir: Path, base_output_dir: Path):
    """Runs segmentation and filtering for one configuration on the dataset."""
    run_name = create_run_name(config)
    run_output_dir = base_output_dir / run_name
    logging.info(f"--- Starting Experiment Run: {run_name} ---")
    logging.info(f"Output directory: {run_output_dir}")

    run_output_dir.mkdir(parents=True, exist_ok=True)

    # Save the exact config used for this run
    try:
        with open(run_output_dir / "params.json", 'w') as f:
            json.dump(config, f, indent=4)
    except Exception as e:
        logging.error(f"Failed to save params.json: {e}")

    # --- Initialize Components --- 
    try:
        segmenter = Segmenter(model_name=config.get('segmenter_model', DEFAULT_CONFIG['segmenter_model']))
        # TODO: Pre-load YOLO model here if apply_clutter_filter is True for efficiency
    except Exception as e:
        logging.exception(f"Failed to initialize segmenter: {e}")
        return
    # ---------------------------

    # --- Find Input Images --- 
    image_paths = sorted(list(input_dir.glob('*.jpg')) + list(input_dir.glob('*.png')) + list(input_dir.glob('*.jpeg')))
    num_images = len(image_paths)
    if num_images == 0:
        logging.warning(f"No images found in {input_dir}. Stopping run.")
        return
    logging.info(f"Found {num_images} images to process.")
    # -------------------------

    # --- Process Images --- 
    success_count = 0
    fail_count = 0
    filter_reject_count = 0

    for i, img_path in enumerate(image_paths):
        logging.info(f"Processing {i+1}/{num_images}: {img_path.name}")
        output_basename = img_path.stem

        try:
            # 1. Load Image
            input_pil = load_image(img_path, mode='RGB')
            if input_pil is None:
                logging.warning(f"Skipping - Failed to load image {img_path.name}")
                fail_count += 1
                continue
            input_rgb_np = np.array(input_pil)
            # Ensure BGR for filtering functions expecting OpenCV default
            input_bgr_np = cv2.cvtColor(input_rgb_np, cv2.COLOR_RGB2BGR) 

            # 2. Raw Segmentation
            raw_mask_np = segmenter.segment(
                input_rgb_np, # Segmenter uses RGB PIL internally
                save_intermediate=False # Don't save raw mask from segmenter
                # output_basename=output_basename,
                # intermediate_dir=run_output_dir # Control saving externally
            )
            if raw_mask_np is None:
                logging.warning(f"Skipping - Segmentation failed for {img_path.name}")
                fail_count += 1
                continue
                
            # Save raw mask externally
            save_image(Image.fromarray(raw_mask_np), run_output_dir / f"{output_basename}_mask_raw.png")

            # 3. Filtering Stage
            final_mask_np = raw_mask_np.copy() # Start with raw mask
            filter_passed = True
            filter_reason = ""
            active_filters = []

            if config.get('apply_contour_filter', False):
                active_filters.append(("Contour", filtering.filter_by_contour))
            if config.get('apply_contrast_filter', False):
                active_filters.append(("Contrast", filtering.filter_by_contrast))
            # if config.get('apply_text_filter', False): # Text filter skipped
            #     active_filters.append(("Text", filtering.filter_by_text_overlap))
            if config.get('apply_clutter_filter', False):
                active_filters.append(("Clutter", filtering.filter_by_clutter))

            logging.debug(f"Applying filters: {[name for name, _ in active_filters]}")
            for filter_name, filter_func in active_filters:
                try:
                    # Pass BGR image to filters as they use OpenCV
                    filter_passed, filter_reason = filter_func(input_bgr_np, final_mask_np, config)
                    if not filter_passed:
                        logging.info(f"Rejected by {filter_name}: {filter_reason}")
                        break
                except Exception as filter_e:
                    logging.error(f"Error during {filter_name} filter: {filter_e}", exc_info=True)
                    filter_passed = False
                    filter_reason = f"Error in {filter_name}"
                    break

            if not filter_passed:
                logging.info(f"Filter rejection for {img_path.name}: {filter_reason}. Not saving final mask/visualization.")
                filter_reject_count += 1
                # Optionally save a marker or log file indicating rejection
                continue # Move to next image

            # 4. Morphological Refinement (if enabled)
            if config.get('refine_mask', False):
                try:
                    logging.debug("Applying morphological refinement...")
                    final_mask_np = filtering.refine_morphological(final_mask_np, config)
                except Exception as refine_e:
                    logging.error(f"Skipping - Error during refinement for {img_path.name}: {refine_e}")
                    fail_count += 1
                    continue
            
            # 5. Feather Mask Edges (Optional)
            feather_amount = config.get('edge_feathering_amount', 0.0)
            if feather_amount > 0:
                try:
                    logging.debug(f"Applying edge feathering (sigma={feather_amount})...")
                    # Ensure mask is float32 for GaussianBlur
                    mask_float = final_mask_np.astype(np.float32) / 255.0
                    # Calculate appropriate kernel size (must be odd)
                    k_size = int(6 * feather_amount + 1)
                    if k_size % 2 == 0: k_size += 1 
                    # Apply blur
                    feathered_mask_float = cv2.GaussianBlur(mask_float, (k_size, k_size), feather_amount)
                    # Convert back to uint8
                    final_mask_np = (feathered_mask_float * 255).clip(0, 255).astype(np.uint8)
                    logging.debug("   Mask feathering complete.")
                except Exception as feather_e:
                    logging.warning(f"Failed to apply feathering for {img_path.name}: {feather_e}. Using unfeathered mask.", exc_info=True)
                    # Continue with the unfeathered mask if an error occurs
            else:
                logging.debug("Skipping edge feathering (amount <= 0).")
            
            # 6. Save Final Mask & Visualization
            save_image(Image.fromarray(final_mask_np), run_output_dir / f"{output_basename}_mask_final.png")
            
            # Ensure raw mask is binary 0/255 for vis
            if raw_mask_np.dtype != np.uint8 or set(np.unique(raw_mask_np)) - {0, 255}:
                 _, raw_mask_binary = cv2.threshold(raw_mask_np, 127, 255, cv2.THRESH_BINARY)
            else:
                 raw_mask_binary = raw_mask_np
            # Ensure final mask is binary 0/255 for vis
            if final_mask_np.dtype != np.uint8 or set(np.unique(final_mask_np)) - {0, 255}:
                 _, final_mask_binary = cv2.threshold(final_mask_np, 127, 255, cv2.THRESH_BINARY)
            else:
                 final_mask_binary = final_mask_np
                 
            vis_img = create_visualization(input_bgr_np, raw_mask_binary, final_mask_binary)
            save_image(vis_img, run_output_dir / f"{output_basename}_visualization.png")

            success_count += 1

        except Exception as e:
            logging.exception(f"Unexpected error processing {img_path.name}: {e}")
            fail_count += 1
    # ----------------------

    logging.info(f"--- Experiment Run Finished: {run_name} ---")
    logging.info(f"Successfully processed: {success_count}")
    logging.info(f"Filter rejections: {filter_reject_count}")
    logging.info(f"Other failures: {fail_count}")
    logging.info(f"Results saved to: {run_output_dir}")
    print() # Add space between runs

def main():
    parser = argparse.ArgumentParser(description="Run segmentation experiments.")
    
    # --- Define Command-Line Arguments to Override Config --- 
    # Use project_root calculated earlier for default paths
    parser.add_argument('--input_dir', type=str, default=str(project_root / 'data' / 'images'), help='Input image directory')
    # Change argument name to match orchestrator
    parser.add_argument('--output_base_dir', type=str, default=str(project_root / 'results' / 'segmentation_experiments'), help='Base output directory for experiments')
    
    # Change argument name to match orchestrator
    parser.add_argument('--model', type=str, dest='segmenter_model', # Keep dest for consistency with config keys if needed
                        help=f"Override segmenter_model (default: {DEFAULT_CONFIG['segmenter_model']}) Options: {list(filtering.SUPPORTED_MODELS.keys()) if hasattr(filtering, 'SUPPORTED_MODELS') else 'See segmentation.py'}")
    parser.add_argument('--refine_mask', action=argparse.BooleanOptionalAction, help="Enable/disable morphological refinement")
    
    # Add arguments for morphological parameters
    parser.add_argument('--opening_kernel', type=int, dest='mask_opening_kernel_size', help="Override mask_opening_kernel_size")
    parser.add_argument('--opening_iter', type=int, dest='mask_opening_iterations', help="Override mask_opening_iterations")
    parser.add_argument('--dilation_kernel', type=int, dest='mask_dilation_kernel_size', help="Override mask_dilation_kernel_size")
    parser.add_argument('--dilation_iter', type=int, dest='mask_dilation_iterations', help="Override mask_dilation_iterations")
    
    # Add argument for feathering
    parser.add_argument('--feather_amount', type=float, dest='edge_feathering_amount', help="Override edge_feathering_amount")

    # Add arguments for Closing parameters
    parser.add_argument('--closing_kernel', type=int, dest='mask_closing_kernel_size', help="Override mask_closing_kernel_size for refinement")
    parser.add_argument('--closing_iter', type=int, dest='mask_closing_iterations', help="Override mask_closing_iterations for refinement")

    parser.add_argument('--apply_contrast_filter', action=argparse.BooleanOptionalAction, help="Enable/disable contrast filter")
    # Change argument name to match orchestrator, keep dest for config key consistency
    parser.add_argument('--contrast_threshold', type=float, dest='contrast_filter_threshold',
                         help="Override contrast_filter_threshold")

    parser.add_argument('--apply_clutter_filter', action=argparse.BooleanOptionalAction, help="Enable/disable clutter filter")
    parser.add_argument('--clutter_min_primary_iou', type=float, help="Override clutter_min_primary_iou")
    parser.add_argument('--clutter_max_other_overlap', type=float, help="Override clutter_max_other_overlap")

    parser.add_argument('--apply_contour_filter', action=argparse.BooleanOptionalAction, help="Enable/disable contour filter")
    parser.add_argument('--contour_min_solidity', type=float, help="Override contour_min_solidity")
    # Add other relevant args like kernel sizes, other thresholds etc.

    args = parser.parse_args()

    # --- Load and Override Config --- 
    config = load_config() # Load defaults
    
    # Apply overrides from args if they were provided (are not None)
    # Use vars(args) to iterate through parsed arguments and their values
    for arg_name, value in vars(args).items():
        if value is not None:
            # Map arg names to config keys if they differ (e.g., via dest)
            # or if they are the same
            config_key = arg_name # Default: assume arg name is config key
            # Special handling for args where name != config key (if dest wasn't used or needs mapping)
            # We used dest, so this mapping isn't strictly needed now, but good practice:
            # if arg_name == 'model': 
            #    config_key = 'segmenter_model'
            # elif arg_name == 'contrast_threshold':
            #    config_key = 'contrast_filter_threshold'
            # ... add other mappings if necessary ...
            
            # Check if the key exists in DEFAULT_CONFIG to avoid adding unrelated args
            # (like input_dir, output_base_dir)
            if config_key in DEFAULT_CONFIG or config_key in [
                'mask_opening_kernel_size', 'mask_opening_iterations',
                'mask_dilation_kernel_size', 'mask_dilation_iterations',
                'mask_closing_kernel_size', 'mask_closing_iterations', # Add closing keys
                'edge_feathering_amount',
                'segmenter_model', # Include keys that might be mapped via dest
                'contrast_filter_threshold', 'clutter_min_primary_iou',
                'clutter_max_other_overlap', 'contour_min_solidity',
                'apply_contrast_filter', 'apply_clutter_filter', 'apply_contour_filter', 'refine_mask' 
                # Add any other expected config keys derived from args
            ]:
                 logging.debug(f"Overriding config: '{config_key}' = {value}")
                 config[config_key] = value
    
    # Note: input_dir and output_base_dir are handled separately below

    logging.info("Loaded configuration with overrides:")
    # Pretty print relevant parts of the final config
    print(json.dumps({k: config.get(k) for k in [
        'segmenter_model', 'refine_mask', 
        'mask_opening_kernel_size', 'mask_opening_iterations', 
        'mask_dilation_kernel_size', 'mask_dilation_iterations', # Keep dilation here for now in case config has it
        'mask_closing_kernel_size', 'mask_closing_iterations', # Add closing keys
        'edge_feathering_amount',
        'apply_contrast_filter', 'contrast_filter_threshold', 
        'apply_clutter_filter', 'clutter_min_primary_iou', 'clutter_max_other_overlap', 
        'apply_contour_filter', 'contour_min_solidity'
        # Add other varied keys here
    ] if k in config}, indent=4))
    print("---")
    # -------------------------------

    # --- Run the Experiment --- 
    input_path = Path(args.input_dir)
    # Use the correct argument name here
    output_path = Path(args.output_base_dir) 
    run_single_experiment(config, input_path, output_path)
    # --------------------------

if __name__ == "__main__":
    main() 