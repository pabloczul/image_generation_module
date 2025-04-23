#!/usr/bin/env python3
"""
Command-line interface for the background generation pipeline.
"""

import argparse
import sys
from pathlib import Path
import json
from typing import Union, Tuple, Dict

# Add src directory to Python path to allow importing pipeline components
# This assumes the script is run from the project root or the `scripts` directory.
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.pipeline.main_pipeline import GenerationPipeline
except ImportError as e:
    print(f"Error: Could not import pipeline components. Ensure you are running from the project root or that the 'src' directory is in your PYTHONPATH. Details: {e}")
    sys.exit(1)

def parse_background_spec(spec_str: str) -> Union[str, Path, Tuple, Dict]:
    """
    Parses the background specification string from the command line.
    Tries to interpret as:
    1. Path to an image file.
    2. JSON dictionary for complex specs (e.g., gradient, diffusion).
    3. Comma-separated RGB tuple (e.g., "255,255,255").
    """
    # Try parsing as JSON dictionary first
    try:
        data = json.loads(spec_str)
        if isinstance(data, dict):
            # Basic validation for known types
            if 'type' in data and data['type'] in ['solid', 'gradient', 'diffusion', 'file']:
                 print(f"Parsed background spec as Dict: {data}")
                 return data
            else:
                 print(f"Warning: Parsed JSON dictionary but unknown type: {data.get('type')}. Treating as path.")
    except json.JSONDecodeError:
        pass # Not a valid JSON dict

    # Try parsing as RGB tuple (e.g., "240,240,240")
    parts = spec_str.split(',')
    if len(parts) == 3:
        try:
            rgb = tuple(int(p.strip()) for p in parts)
            if all(0 <= x <= 255 for x in rgb):
                 print(f"Parsed background spec as RGB Tuple: {rgb}")
                 return rgb
            else:
                 print("Warning: Parsed 3 comma-separated values, but not valid RGB [0-255]. Treating as path.")
        except ValueError:
            pass # Not three integers

    # Default: Treat as a file path
    print(f"Treating background spec as Path: {spec_str}")
    return Path(spec_str)

def main():
    parser = argparse.ArgumentParser(description="Generate product images with new backgrounds.")

    parser.add_argument("input_image", type=str,
                        help="Path to the input product image file.")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Path for the output image file (e.g., output/result.png).")
    
    parser.add_argument("-bg", "--background", type=str, required=True,
                        help=("Background specification. Can be: \n"
                              "  - Path to a background image (e.g., 'backgrounds/wood.jpg'). \n"
                              "  - RGB color tuple (e.g., '255,255,255' for white). \n"
                              "  - JSON string for complex types: \n"
                              "    '{\"type\": \"gradient\", \"colors\": [[240,240,240], [220,220,220]], \"direction\": \"vertical\"}' \n"
                              "    '{\"type\": \"diffusion\", \"prompt\": \"A sunny beach background\"}'"))
    
    parser.add_argument("-p", "--prompt", type=str, default=None,
                        help="Text prompt, primarily for 'diffusion' background type.")
    
    # Pipeline configuration overrides
    parser.add_argument("--segmenter_model", type=str, default=None, # Default handled by pipeline config
                        help="Override segmentation model (e.g., u2net, u2netp, silueta).")
    parser.add_argument("--diffusion_cfg", type=str, default=None,
                        help=("JSON string with configuration overrides for DiffusionGenerator, e.g., "
                              "'{\"device\": \"cuda\", \"num_inference_steps\": 50}'."))
    # parser.add_argument("--config_file", type=str, default=None,
    #                     help="Path to a custom pipeline configuration file (YAML/JSON - TODO).")

    # New flag for saving intermediate masks
    parser.add_argument("--save-intermediate-masks", action="store_true",
                        help="Save raw and refined segmentation masks to the intermediate directory specified in config.")

    args = parser.parse_args()

    # Parse background spec
    try:
        bg_spec = parse_background_spec(args.background)
    except Exception as e:
         print(f"Error parsing background specification '{args.background}': {e}", file=sys.stderr)
         sys.exit(1)

    # Parse diffusion config if provided
    diffusion_options = None
    if args.diffusion_cfg:
        try:
             diffusion_options = json.loads(args.diffusion_cfg)
             if not isinstance(diffusion_options, dict):
                  raise ValueError("diffusion_cfg must be a JSON dictionary.")
             print(f"Using custom diffusion config: {diffusion_options}")
        except (json.JSONDecodeError, ValueError) as e:
             print(f"Error parsing --diffusion_cfg: {e}. Please provide a valid JSON dictionary string.", file=sys.stderr)
             sys.exit(1)
             
    # TODO: Load base pipeline config from file if specified
    # config_path = args.config_file 
    config_path = None

    # Initialize pipeline, passing overrides from CLI args
    try:
        pipeline = GenerationPipeline(
            config_path=config_path,
            segmenter_model=args.segmenter_model, # Pass None if not specified, pipeline uses config default
            diffusion_cfg_overrides=diffusion_options, # Pass None if not specified
            # Pass the value of the flag (True if present, False otherwise)
            # Set to None if flag not present, so pipeline uses config default
            save_intermediate_masks_override=args.save_intermediate_masks if args.save_intermediate_masks else None
        )
    except Exception as e:
        print(f"Error initializing generation pipeline: {e}", file=sys.stderr)
        sys.exit(1)

    # Run processing
    print(f"\nProcessing {args.input_image} -> {args.output}")
    success = pipeline.process_image(
        image_path=args.input_image,
        output_path=args.output,
        background_spec=bg_spec,
        prompt=args.prompt
    )

    if success:
        print("\nProcessing finished successfully.")
        sys.exit(0)
    else:
        print("\nProcessing failed.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 