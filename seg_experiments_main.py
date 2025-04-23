"""
Orchestrator script for running segmentation experiments.

This script automatically executes 'run_segmentation_experiments.py' 
multiple times with different parameter combinations defined in the 
PARAMETER_GRID below. It iterates through all unique combinations
and launches the experiment script for each.
"""

import subprocess
import itertools
import os
from typing import Dict, List, Any, Tuple, Union

# --- Configuration ---

# <<< --- ADDED CONTROL FLAG --- >>>
# Set to True to only test different models (refinement disabled).
# Set to False to test the full grid including morphological variations.
TEST_ONLY_MODELS = False 
# <<< -------------------------- >>>

# Define the path to the experiment script
# Assumes this script is run from the project root
EXPERIMENT_SCRIPT_PATH = "scripts/run_segmentation_experiments.py" 

# Define the FULL grid of parameters to test when TEST_ONLY_MODELS is False
FULL_PARAMETER_GRID: Dict[str, List[Union[str, int, float, None]]] = {
    # Core Segmentation & Refinement Toggle
    "--model": ["isnet-general-use"], #"u2net", "u2netp", "isnet-general-use", "silueta"], 
    "--refine_mask": [True], # Force refinement ON for this strategy
    
    # --- Morphological Parameters for Opening -> Closing --- 
    # Stage 1: Gentle Opening (Fixed Parameters)
    "--opening_kernel": [5], # Keep small to remove noise without eroding
    "--opening_iter": [1],    # Keep minimal
    
    # Stage 2: Closing for Hole Filling (Vary Parameters)
    "--closing_kernel": [15, 7], # Test different kernel sizes for hole filling
    "--closing_iter": [1, 4],    # Test iterations for hole filling
    
    # Remove standalone Dilation parameters as Closing handles hole filling
    # "--dilation_kernel": [X], 
    # "--dilation_iter": [Y], 
    # --------------------------------------------------------
    
    # Edge Treatment (Keep if desired after Closing)
    "--feather_amount": [0.0, 2.0], 
    
    # --- Activate Filters --- 
    # Set filter flags to True to enable them
    "--apply_contrast_filter": [False],
    "--contrast_threshold": [5.0, 10.0], # Vary threshold only when filter is active
    
    "--apply_clutter_filter": [False],
    # TODO: Add variations for clutter params if needed, e.g.:
    # "--clutter_min_primary_iou": [0.6, 0.7],
    # "--clutter_max_other_overlap": [0.1, 0.2],
    
    "--apply_contour_filter": [False],
    # TODO: Add variations for contour params if needed, e.g.:
    # "--contour_min_solidity": [0.8, 0.85],
    # ------------------------

    # Required arguments for the script (Use argument names expected by script)
    "--input_dir": ["data/test_images"], 
    "--output_base_dir": ["results/seg_experiments_opening_closing"], # New output dir name
}

# --- Select the Grid to Use --- 
if TEST_ONLY_MODELS:
    print("WARNING: TEST_ONLY_MODELS is True, but the grid is configured for Opening+Closing refinement. Set TEST_ONLY_MODELS=False to run the intended experiment.")
    # Keep the simplified grid definition here as a fallback or for other tests
    PARAMETER_GRID = {
        "--model": FULL_PARAMETER_GRID["--model"], # Iterate through models
        "--refine_mask": [None],  # Force refinement off
        # Add filter flags if you want them active in this mode too:
        # "--apply_contrast_filter": [True],
        # "--contrast_threshold": [10.0], # Fixed value if not varying
        # "--apply_clutter_filter": [True],
        # "--apply_contour_filter": [True],
        # Make sure required args are present
        "--input_dir": FULL_PARAMETER_GRID["--input_dir"],
        "--output_base_dir": [f"{FULL_PARAMETER_GRID['--output_base_dir'][0]}_models_only"], # Use different output dir
        # Ensure necessary default/fixed values for params expected by the script are passed
        # These won't be varied, but might be needed by run_segmentation_experiments.py
        # Check the argparse defaults in that script. If a default is None and the arg is 
        # required for config, you might need to add it here with a fixed value.
        # Example: if contrast_threshold had no default in run_script and was needed:
        # "--contrast_threshold": [10.0], 
    }
else:
    print("INFO: Running with FULL parameter grid configured for Opening->Closing refinement.")
    PARAMETER_GRID = FULL_PARAMETER_GRID
# -----------------------------

# --- Helper Functions ---

def generate_parameter_combinations(
    grid: Dict[str, List[Any]]
) -> List[Dict[str, Any]]:
    """Generates all unique combinations from the parameter grid."""
    keys = list(grid.keys())
    value_lists = list(grid.values())
    
    combinations = []
    for values_tuple in itertools.product(*value_lists):
        combination = dict(zip(keys, values_tuple))
        combinations.append(combination)
    return combinations

def build_command(
    base_command: List[str], 
    params: Dict[str, Any]
) -> List[str]:
    """Constructs the command list for subprocess."""
    command = base_command[:] # Copy base command
    # Use underscores in keys to match PARAMETER_GRID keys
    boolean_optional_args = {
         "--refine_mask",
         "--apply_contrast_filter", 
         "--apply_clutter_filter",
         "--apply_contour_filter"
         # Add any other BooleanOptionalAction arg keys from PARAMETER_GRID
    }
    
    for arg, value in params.items():
        # Check if the current argument key is one of our boolean flags
        if arg in boolean_optional_args:
            # Handle boolean flags based on their True/False/None value
            if value is True:
                 # If True, add the positive flag (e.g., --refine_mask)
                 command.append(arg)
            elif value is None or value is False:
                 # If False or None, add the negative flag (e.g., --no-refine-mask)
                 if arg.startswith('--'):
                      command.append(f"--no-{arg[2:]}") 
            # Do NOT append the value ('True'/'False') for boolean flags
        elif value is None:
            # Skip other arguments if their value is None
            continue
        else:
            # This block now only handles non-boolean arguments
            # Add the argument name and its string value
            command.append(arg)
            command.append(str(value))
    return command

# --- Main Execution ---

def main():
    """Runs the main experiment orchestration loop."""
    print("Starting segmentation experiments orchestration...")
    
    if not os.path.exists(EXPERIMENT_SCRIPT_PATH):
        print(f"Error: Experiment script not found at '{EXPERIMENT_SCRIPT_PATH}'")
        return

    combinations = generate_parameter_combinations(PARAMETER_GRID)
    total_combinations = len(combinations)
    
    print(f"Generated {total_combinations} parameter combinations.")
    
    base_command = ["python", EXPERIMENT_SCRIPT_PATH]

    for i, params in enumerate(combinations):
        print(f"--- Running Combination {i+1}/{total_combinations} ---")
        
        command = build_command(base_command, params)
        
        print(f"Executing: {' '.join(command)}")
        
        try:
            # Execute the command sequentially
            result = subprocess.run(command, check=True, text=True, capture_output=True)
            print(f"Successfully completed combination {i+1}.")
            # Optional: Print stdout/stderr if needed for debugging
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            
        except FileNotFoundError:
            print(f"Error: 'python' command not found. Is Python installed and in PATH?")
            break # Stop if python is not found
        except subprocess.CalledProcessError as e:
            print(f"Error executing command for combination {i+1}:")
            print(f"Command: {' '.join(e.cmd)}")
            print(f"Return Code: {e.returncode}")
            print(f"STDOUT:{e.stdout}")
            print(f"STDERR:{e.stderr}")
            print("Stopping experiment run due to error.")
            break # Stop the loop if one experiment fails
        except Exception as e:
            print(f"An unexpected error occurred during execution of combination {i+1}: {e}")
            print("Stopping experiment run due to unexpected error.")
            break # Stop on other unexpected errors

    print("--- Experiment orchestration finished. ---")

if __name__ == "__main__":
    main() 