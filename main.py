import sys
import time
from pathlib import Path
from typing import Dict, Any
import json # Needed for potential config overrides if we were using them

# Add src directory to Python path to allow importing pipeline components
# Assumes this script is in the project root directory
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    print(f"Added {project_root} to sys.path")

try:
    from src.pipeline.main_pipeline import GenerationPipeline
    from src.utils.data_io import get_image_paths
    print("Pipeline components imported successfully.")
except ImportError as e:
    print(f"Error: Could not import pipeline components: {e}", file=sys.stderr)
    print("Please ensure you are running this script from the project root", file=sys.stderr)
    print("and have installed all requirements.", file=sys.stderr)
    sys.exit(1)

# --- Configuration ---
INPUT_IMAGE_DIR = project_root / "data" / "images"
OUTPUT_DIR = project_root / "results" / "diffusion_run_outputs"

# Background Specifications
# Using defaults from config for gradient, defining diffusion spec
# You can customize the gradient here if needed, e.g.:
# GRADIENT_SPEC = {'type': 'gradient', 'colors': [[255,255,255], [200,200,220]], 'direction': 'radial'}
GRADIENT_SPEC = {'type': 'gradient'} # Uses default gradient from config

DIFFUSION_SPEC = {'type': 'diffusion'}
# A generic prompt for diffusion backgrounds
DIFFUSION_PROMPT = "Clean, neutral, photorealistic, professional studio background, soft lighting, light gray background, high quality product shot"

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Output directory: {OUTPUT_DIR.resolve()}")

# --- Main Execution ---
def run_batch_processing():
    start_run_time = time.time()

    # --- Initialize Pipeline ---
    # IMPORTANT: Explicitly enabling diffusion for this run.
    # Assumes user has necessary hardware and models installed/downloadable.
    print("\nInitializing Generation Pipeline (Diffusion Explicitly Enabled)...")
    pipeline: GenerationPipeline | None = None
    try:
        # Explicitly enable diffusion.
        pipeline = GenerationPipeline(diffusion_enabled=True)
        if not pipeline.diffusion_generator:
             # Keep the warning but don't exit, allow summary to show 0 successes
             print("\n--- WARNING ---")
             print("Pipeline initialized, but DiffusionGenerator failed to load.")
             print("Diffusion backgrounds will NOT be generated.")
             print("Check error messages above for details (GPU/CUDA issues, missing models, etc.).")
             print("---------------\n")
    except Exception as e:
        print(f"\nFatal Error initializing pipeline: {e}", file=sys.stderr)
        print("Cannot proceed.", file=sys.stderr)
        sys.exit(1)

    # --- Get Input Images ---
    print(f"\nLooking for images in: {INPUT_IMAGE_DIR}")
    input_image_paths = get_image_paths(INPUT_IMAGE_DIR, recursive=False)

    if not input_image_paths:
        print(f"Error: No images found in {INPUT_IMAGE_DIR}. Please add images or run download script.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(input_image_paths)} images to process.")

    # --- Process Each Image ---
    success_count_diff = 0
    fail_count_diff = 0
    quality_issues_count = 0

    for i, img_path in enumerate(input_image_paths):
        print(f"\n--- Processing Image {i+1}/{len(input_image_paths)}: {img_path.name} ---")
        base_name = img_path.stem

        # 1. Assess Input Quality (Optional, kept for info)
        print("   Assessing input quality...")
        quality_assessment = pipeline.assessor.assess_quality(img_path)
        quality_ok = quality_assessment.get('is_processable', False)
        print(f"   Quality Assessment: {quality_assessment.get('message', 'N/A')}")
        if not quality_ok:
            print(f"   -> Input quality below threshold. Processing may yield suboptimal results.")
            quality_issues_count += 1
        else:
            print(f"   -> Input quality sufficient.")

        # 2. Process ONLY with Diffusion Background (if generator initialized)
        if pipeline.diffusion_generator:
            print("   Processing with Diffusion Background...")
            output_path_diff = OUTPUT_DIR / f"{base_name}_diffusion.png"
            try:
                success_diff = pipeline.process_image(
                    image_path=img_path,
                    output_path=output_path_diff,
                    background_spec=DIFFUSION_SPEC,
                    prompt=DIFFUSION_PROMPT
                )
                if success_diff:
                    print(f"   -> Diffusion output saved: {output_path_diff.name}")
                    success_count_diff += 1
                else:
                    print(f"   -> Diffusion processing FAILED.")
                    fail_count_diff += 1
            except Exception as e:
                 print(f"   -> UNEXPECTED ERROR during Diffusion processing: {e}")
                 fail_count_diff += 1
        else:
             # Log skipping only if generator wasn't initialized
             print("   Skipping Diffusion (Generator not available).")
             fail_count_diff += 1

    # --- Final Summary ---
    end_run_time = time.time()
    total_time = end_run_time - start_run_time
    print("\n--- Batch Processing Summary ---")
    print(f"Total images processed: {len(input_image_paths)}")
    print(f"Input images with quality warnings: {quality_issues_count}")
    print("-" * 20)
    print("Diffusion Backgrounds:")
    print(f"  Successfully generated: {success_count_diff}")
    print(f"  Failed/Skipped:       {fail_count_diff}")
    print("-" * 20)
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Results saved in: {OUTPUT_DIR.resolve()}")
    print("--- End Summary ---")

if __name__ == "__main__":
    run_batch_processing()