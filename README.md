# Product Background Generation Pipeline

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
<!-- Add other badges if applicable (e.g., License, Build Status) -->

This project provides a modular Python pipeline for automatically replacing the background of product images with various generated or user-provided backgrounds.

## Overview

The primary goal is to take an input image of a product (e.g., furniture) on a simple background, automatically segment the product, and place it onto a new background. This new background can be a solid color, a gradient, an existing image file, or generated using diffusion models (like Stable Diffusion + ControlNet).

The project has been refactored from initial scripts into a production-oriented structure with a core library (`src`), command-line interface (`scripts`), and support for Docker deployment.

## Features

*   **Modular Architecture:** Codebase organized into logical components (segmentation, quality assessment, background generation, pipeline orchestration, utilities).
*   **Foreground Segmentation:** Uses the `rembg` library (`u2net` model by default) to automatically isolate the foreground product.
*   **Background Types:**
    *   **Solid Color:** Generate backgrounds with a specified RGB color.
    *   **Gradient:** Generate linear (vertical/horizontal) or radial gradient backgrounds.
    *   **Image File:** Use any existing image file as a background.
    *   **Diffusion (Optional):** Leverage Stable Diffusion with ControlNet (Inpainting + Segmentation/Canny) to generate context-aware backgrounds based on text prompts.
*   **Image Quality Assessment:** Includes utilities to check input image resolution, blur, and contrast (currently simplified in main pipeline).
*   **Configurable Mask Refinement:** Applies morphological opening and dilation (tunable kernel sizes/iterations via `src/config.py`) to improve the segmentation mask.
*   **Configurable Edge Feathering:** Option to apply Gaussian blur to mask edges for smoother compositing (tunable sigma via `src/config.py`).
*   **Configurable Drop Shadow:** Option to add a soft drop shadow with configurable offset, blur (sigma), opacity, and color via `src/config.py`.
*   **Command-Line Interface:** Easy-to-use CLI (`scripts/generate.py`) for processing images.
*   **Docker Support:** Includes `Dockerfile` for building and running the application in a containerized environment (CPU-based by default).
*   **Configuration:** Centralized default settings in `src/config.py`.
*   **Intermediate Mask Saving:** Optional flag (`--save-intermediate-masks` or via config) to save raw and refined masks for debugging.

## Project Structure

```
├── data/
│   ├── images/          # Input product images
│   ├── backgrounds/     # Generated/Downloaded background images
│   └── metadata/        # CSV files or other metadata
├── docs/                # Project documentation
├── notebooks/
│   └── pipeline_demonstration.ipynb # Example usage notebook
├── results/
│   ├── main_run_outputs/ # Default output dir for main.py
│   ├── intermediate_masks/ # Default output dir for intermediate masks
│   └── notebook_demonstration/ # Default output dir for notebook
├── scripts/
│   ├── download_data.py # Script to download sample images/backgrounds
│   └── generate.py      # Main CLI script for processing images
├── src/
│   ├── __init__.py
│   ├── background/      # Background generation/loading/utils
│   ├── image/           # Segmentation, quality assessment
│   ├── models/          # Diffusion model wrappers
│   ├── pipeline/        # Main pipeline orchestration
│   ├── utils/           # Data I/O, visualization
│   └── config.py        # Configuration settings
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_background/
│   ├── test_image/
│   ├── test_pipeline/
│   └── test_utils/
├── .dockerignore        # Files ignored by Docker build
├── Dockerfile           # Defines the Docker image
├── main.py              # Example batch processing script
├── requirements.txt     # Python dependencies
├── PROJECT_STATUS.md    # Detailed status tracking
└── README.md            # This file
```

## Setup

Follow these steps to set up the project locally.

1.  **Clone the Repository (if applicable):**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Create and Activate Virtual Environment:**
    ```bash
    # Create environment
    python -m venv .venv

    # Activate (Linux/macOS/Git Bash)
    source .venv/bin/activate
    # OR Activate (Windows CMD/PowerShell)
    # .\.venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    Install packages from `requirements.txt` and explicitly install the correct PyTorch version for your system (CPU version recommended for broad compatibility if GPU is not set up).
    ```bash
    # Install general requirements
    pip install -r requirements.txt

    # Install PyTorch (CPU example)
    pip install torch==2.1.0+cpu torchvision==0.16.0+cpu torchaudio==2.1.0+cpu --index-url https://download.pytorch.org/whl/cpu
    # (For GPU, visit pytorch.org to find the correct command for your CUDA version)
    ```

4.  **Download Sample Data (Optional):**
    Run the download script to get sample product images and backgrounds.
    ```bash
    python scripts/download_data.py
    # Use --skip_products or --skip_backgrounds if you only need one
    ```

5.  **Model Downloads (Automatic):**
    The necessary `rembg` and `diffusers` models will be downloaded automatically on first use when running the pipeline (requires internet connection).

## Usage

Make sure your virtual environment is activated and run commands from the project root directory.

### Command-Line Interface (CLI)

The main tool is `scripts/generate.py`.

**Syntax:**
```bash
python scripts/generate.py <input_image> -o <output_image> -bg <background_spec> [options]
```

**Arguments:**

*   `<input_image>`: (Required) Path to the input product image.
*   `-o <output_image>` / `--output <output_image>`: (Required) Path for the output image (e.g., `results/output.png`). Extension determines format.
*   `-bg <background_spec>` / `--background <background_spec>`: (Required) Defines the background. Examples:
    *   File Path: `data/backgrounds/some_texture.jpg`
    *   RGB Tuple: `"255,255,255"` (Use quotes)
    *   JSON (Solid): `'{"type": "solid", "color": [100, 150, 200]}'` (Use single quotes around JSON, double quotes inside)
    *   JSON (Gradient): `'{"type": "gradient", "colors": [[240,240,240], [220,220,220]], "direction": "vertical"}'`
    *   JSON (Diffusion): `'{"type": "diffusion"}'` (Use with `-p` option)
    *   JSON (File): `'{"type": "file", "path": "data/backgrounds/image.png"}'`
*   `-p <prompt>` / `--prompt <prompt>`: Optional text prompt for diffusion backgrounds.
*   `--segmenter_model <model_name>`: Optional. Override segmentation model (e.g., `u2net`, `u2netp`, `silueta`). If not provided, uses the default from `src/config.py`.
*   `--diffusion_cfg <json_string>`: Optional. JSON string with overrides for DiffusionGenerator settings (e.g., `'{"device": "cuda", "num_inference_steps": 50}'`). Merged with defaults from `src/config.py`.
*   `--save-intermediate-masks`: Optional flag. If present, saves raw and refined masks to the directory specified by `intermediate_mask_dir` in `src/config.py`. Overrides the `save_intermediate_masks` setting in the config file.

**Examples:**

```bash
# Solid white background
python scripts/generate.py data/images/4447872.jpg -o results/output_solid.png -bg "255,255,255"

# Gradient background via JSON, save intermediate masks
python scripts/generate.py data/images/4447872.jpg -o results/output_gradient.png -bg '{"type": "gradient", "colors": [[230,240,255],[200,210,230]]}' --save-intermediate-masks

# Background from image file, override segmenter model
python scripts/generate.py data/images/4447872.jpg -o results/output_filebg.jpg -bg data/backgrounds/wood_texture.jpg --segmenter_model silueta

# Diffusion background (if enabled/available)
# python scripts/generate.py data/images/4447872.jpg -o results/output_diffusion.png -bg '{"type": "diffusion"}' -p "Minimalist light grey studio setting"
```

### Batch Processing Script

The `main.py` script provides an example of processing all images in the `data/images` directory with both diffusion (if available) and gradient backgrounds. Edit the script if needed and run:

```bash
python main.py
```
Outputs will be saved in `results/main_run_outputs/`.

### Library Usage (Python/Notebooks)

You can import and use the `GenerationPipeline` class directly in your Python scripts or notebooks. See `notebooks/pipeline_demonstration.ipynb` for detailed examples.

```python
from src.pipeline.main_pipeline import GenerationPipeline

# Initialize (defaults or specify config/overrides)
# pipeline = GenerationPipeline(diffusion_enabled=True) 
pipeline = GenerationPipeline(
    save_intermediate_masks_override=True # Example override
)

input_img = "data/images/your_image.jpg"
output_img = "results/my_output.png"

# Example: Solid background
bg_spec_solid = (150, 180, 200)
success = pipeline.process_image(input_img, output_img, bg_spec_solid)

# Example: Gradient background
bg_spec_gradient = {'type': 'gradient', 'colors': [(255,255,255), (200,200,200)]}
success = pipeline.process_image(input_img, "results/my_output_gradient.png", bg_spec_gradient)

print(f"Processing {'succeeded' if success else 'failed'}.")
```

### Docker Usage

A `Dockerfile` is provided for building and running the application in a container (CPU by default).

1.  **Build the Image:**
    From the project root directory:
    ```bash
    docker build -t product-background-gen . 
    ```

2.  **Run the Container:**
    Use the CLI via the container's entrypoint. You need to mount volumes for data input and results output.
    ```bash
    docker run --rm \
      -v ./data:/app/data \
      -v ./results:/app/results \
      product-background-gen \
      data/images/YOUR_IMAGE.jpg \
      -o results/docker_output.png \
      -bg "200,220,255"
    ```
    *   Replace `YOUR_IMAGE.jpg` with an actual image filename in `./data/images/`.
    *   `--rm`: Automatically remove the container when it exits.
    *   `-v ./data:/app/data`: Mounts your local `data` directory to `/app/data` inside the container.
    *   `-v ./results:/app/results`: Mounts your local `results` directory to `/app/results` inside the container.
    *   The arguments after the image name are passed to the `scripts/generate.py` entrypoint.

## Configuration

Default settings for the pipeline are defined in `src/config.py` (`DEFAULT_CONFIG` dictionary). You can modify this file directly for persistent changes, or provide overrides during `GenerationPipeline` initialization (library usage) or via specific CLI arguments (`scripts/generate.py`).

Key configurable sections include:

*   **Input Image Processing:** Minimum resolution checks, blur/contrast thresholds.
*   **Segmentation:** Default `rembg` model (`segmenter_model`), device (`segmentation_device`).
*   **Mask Refinement:** `refine_mask` (bool), `mask_opening_kernel_size`, `mask_opening_iterations`, `mask_dilation_kernel_size`, `mask_dilation_iterations`.
*   **Compositing:** `add_shadow` (bool), `edge_feathering_amount` (Gaussian blur sigma for mask edges, 0 disables).
*   **Shadow:** `shadow_offset_x`, `shadow_offset_y`, `shadow_blur_sigma` (softness), `shadow_opacity`, `shadow_color`.
*   **Background Generation:** Default settings for diffusion models (`diffusion_model_id`, `diffusion_controlnet_model_id`, `diffusion_device`, inference steps, scales).
*   **Output:** Default output directories (`default_output_dir`, `intermediate_mask_dir`), `save_intermediate_masks` (bool, can be overridden by CLI flag).

*(Note: Loading configuration from external YAML/JSON files is planned but not yet implemented in `src.config.load_config`)*.

## Testing

The project uses `pytest`. Ensure you have installed the test dependencies (`pytest` is included in `requirements.txt`).

To run the tests, navigate to the project root directory and run:

```bash
pytest
```
Or:
```bash
python -m pytest
```
Use `-v` for more verbose output.

## Status & Future Work

This project has implemented the core refactoring, modularization, main features, CLI, and basic Docker support.

**Pending / Potential Improvements:**

*   **Formal Testing:** Expand unit and integration test coverage in the `tests/` directory.
*   **Configuration Loading:** Implement loading of settings from external YAML/JSON files in `src/config.py`.
*   **Error Handling:** Add more specific error handling and user feedback.
*   **Performance Optimization:** Profile and optimize bottlenecks, especially in image processing and model inference.
*   **GPU Support (Docker):** Provide an alternative Dockerfile or build arguments for GPU-enabled images.
*   **Security Considerations:** Review and implement security best practices if handling sensitive data or deploying publicly.
*   **Advanced Features:** Explore more sophisticated shadow generation, perspective warping, or other image enhancement techniques.

## License

This project is licensed under the MIT License. See the LICENSE file for details (if one exists). 