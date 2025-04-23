"""
Visualization utilities using Matplotlib for displaying images and results 
from the product background generation pipeline.

Requires `matplotlib` to be installed.
"""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Union, Optional, Tuple, Sequence
import warnings

try:
    import matplotlib.pyplot as plt
    import matplotlib.figure
    import matplotlib.axes
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("Matplotlib not found. Visualization functions will not be available. "
                  "Install with: pip install matplotlib")
    # Define dummy types if needed for type hinting without errors
    plt = None
    matplotlib = None 


# --- Helper to handle image input ---

def _to_numpy(image: Union[np.ndarray, Image.Image]) -> Optional[np.ndarray]:
    """Converts PIL Image to NumPy array, returns NumPy array as is."""
    if isinstance(image, Image.Image):
        # Convert RGBA to RGB for standard display
        if image.mode == 'RGBA':
            return np.array(image.convert('RGB'))
        return np.array(image)
    elif isinstance(image, np.ndarray):
        # Handle potential float arrays from processing
        if image.dtype != np.uint8:
             if np.issubdtype(image.dtype, np.floating):
                 # Assume range [0, 1], scale to [0, 255]
                 if np.max(image) <= 1.0 and np.min(image) >= 0.0:
                      image = (image * 255).astype(np.uint8)
                 else: # Clip if outside [0, 255]
                      image = np.clip(image, 0, 255).astype(np.uint8)
             else:
                 # Attempt direct conversion for other types
                 try: image = image.astype(np.uint8)
                 except ValueError:
                     warnings.warn(f"Could not safely convert NumPy array of type {image.dtype} to uint8 for display.")
                     return None
        return image
    else:
        warnings.warn(f"Unsupported image type for display: {type(image)}")
        return None

# --- Visualization Functions ---

def display_image_grid(
    images: Sequence[Union[np.ndarray, Image.Image]], 
    titles: Optional[Sequence[str]] = None, 
    rows: Optional[int] = None, 
    cols: Optional[int] = None, 
    figsize: Tuple[float, float] = (15, 10),
    save_path: Optional[Union[str, Path]] = None
) -> Optional[matplotlib.figure.Figure]:
    """
    Displays a grid of images using Matplotlib.

    Args:
        images: A sequence (list, tuple) of images (NumPy arrays or PIL Images).
        titles: Optional sequence of titles corresponding to the images.
        rows: Optional number of rows in the grid. Auto-calculated if None.
        cols: Optional number of columns in the grid. Auto-calculated if None.
        figsize: Figure size (width, height) in inches.
        save_path: Optional path (str or Path) to save the figure.

    Returns:
        Optional[matplotlib.figure.Figure]: The Matplotlib Figure object, or None if failed.
    """
    if not MATPLOTLIB_AVAILABLE:
        warnings.warn("Matplotlib not available, cannot display image grid.")
        return None
        
    if not images:
        warnings.warn("No images provided to display_image_grid.")
        return None

    n_images = len(images)
    if titles and len(titles) != n_images:
        warnings.warn(f"Number of titles ({len(titles)}) does not match number of images ({n_images}). Titles will be truncated or ignored.")
        titles = titles[:n_images]

    # Calculate grid dimensions
    if rows is None and cols is None:
        cols = min(4, n_images)  # Default max 4 columns
        rows = (n_images + cols - 1) // cols
    elif rows is None:
        rows = (n_images + cols - 1) // cols
    elif cols is None:
        cols = (n_images + rows - 1) // rows
    
    if rows <= 0 or cols <= 0:
        warnings.warn(f"Invalid grid dimensions calculated ({rows}x{cols}). Cannot display grid.")
        return None

    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False) # squeeze=False ensures axes is always 2D array

    for idx, img_input in enumerate(images):
        if idx >= rows * cols:
            break # Stop if we exceed calculated grid size

        row_idx, col_idx = idx // cols, idx % cols
        ax = axes[row_idx, col_idx]
        
        img_np = _to_numpy(img_input)
        
        if img_np is not None:
            # Determine cmap based on dimensions
            if img_np.ndim == 2 or (img_np.ndim == 3 and img_np.shape[2] == 1):
                cmap = 'gray'
            else:
                cmap = None # Default for RGB
            ax.imshow(img_np, cmap=cmap)
            if titles and idx < len(titles):
                ax.set_title(titles[idx])
        else:
             # Optionally display a placeholder or message if conversion failed
             ax.text(0.5, 0.5, 'Invalid Image', ha='center', va='center')
             if titles and idx < len(titles):
                  ax.set_title(f"{titles[idx]} (Error)")
             
        ax.axis('off')

    # Hide unused subplots
    for idx in range(n_images, rows * cols):
        row_idx, col_idx = idx // cols, idx % cols
        axes[row_idx, col_idx].axis('off')

    plt.tight_layout()
    
    if save_path:
        try:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path)
            print(f"Image grid saved to: {save_path}")
        except Exception as e:
            warnings.warn(f"Failed to save image grid to {save_path}: {e}")
            
    # plt.show() # Typically called outside the function if interactive display needed
    return fig

def compare_images(
    images: Sequence[Union[np.ndarray, Image.Image]], 
    titles: Optional[Sequence[str]] = None, 
    figsize: Tuple[float, float] = (15, 5),
    main_title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None
) -> Optional[matplotlib.figure.Figure]:
    """
    Displays a sequence of images side-by-side for comparison.

    Args:
        images: Sequence of images (NumPy arrays or PIL Images) to compare.
        titles: Optional sequence of titles for each image.
        figsize: Figure size (width, height).
        main_title: Optional main title for the entire figure.
        save_path: Optional path to save the figure.

    Returns:
        Optional[matplotlib.figure.Figure]: The Matplotlib Figure object, or None if failed.
    """
    if not MATPLOTLIB_AVAILABLE:
        warnings.warn("Matplotlib not available, cannot compare images.")
        return None
        
    if not images:
        warnings.warn("No images provided to compare_images.")
        return None
        
    n_images = len(images)
    if titles and len(titles) != n_images:
        warnings.warn(f"Number of titles ({len(titles)}) does not match number of images ({n_images}). Titles will be truncated or ignored.")
        titles = titles[:n_images]

    fig, axes = plt.subplots(1, n_images, figsize=figsize, squeeze=False) # Ensure axes is always 2D (1 row)

    for idx, img_input in enumerate(images):
        ax = axes[0, idx]
        img_np = _to_numpy(img_input)
        
        if img_np is not None:
             if img_np.ndim == 2 or (img_np.ndim == 3 and img_np.shape[2] == 1):
                 cmap = 'gray'
             else:
                 cmap = None
             ax.imshow(img_np, cmap=cmap)
             if titles and idx < len(titles):
                 ax.set_title(titles[idx])
        else:
            ax.text(0.5, 0.5, 'Invalid Image', ha='center', va='center')
            if titles and idx < len(titles):
                 ax.set_title(f"{titles[idx]} (Error)")
                 
        ax.axis('off')

    if main_title:
        fig.suptitle(main_title, fontsize=16)
        plt.subplots_adjust(top=0.9) # Adjust layout to make space for title
        
    plt.tight_layout(rect=[0, 0, 1, 0.95] if main_title else None) # Adjust for suptitle

    if save_path:
        try:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path)
            print(f"Image comparison saved to: {save_path}")
        except Exception as e:
            warnings.warn(f"Failed to save image comparison to {save_path}: {e}")
            
    return fig

def plot_histogram(
    image: Union[np.ndarray, Image.Image],
    title: str = "Color Histogram",
    figsize: Tuple[float, float] = (8, 5),
    save_path: Optional[Union[str, Path]] = None
) -> Optional[matplotlib.figure.Figure]:
    """
    Plots the color histogram of an image (RGB or Grayscale).

    Args:
        image: Input image (NumPy array or PIL Image).
        title: Title for the plot.
        figsize: Figure size.
        save_path: Optional path to save the figure.

    Returns:
        Optional[matplotlib.figure.Figure]: The Matplotlib Figure object, or None if failed.
    """
    if not MATPLOTLIB_AVAILABLE:
        warnings.warn("Matplotlib not available, cannot plot histogram.")
        return None
        
    img_np = _to_numpy(image)
    if img_np is None:
        warnings.warn("Cannot plot histogram: Invalid image input.")
        return None

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if img_np.ndim == 3 and img_np.shape[2] == 3: # RGB
        colors = ('r', 'g', 'b')
        labels = ('Red', 'Green', 'Blue')
        for i, color in enumerate(colors):
            # Calculate histogram for channel i
            hist = cv2.calcHist([img_np], [i], None, [256], [0, 256])
            ax.plot(hist, color=color, label=labels[i], alpha=0.8)
        ax.legend()
        ax.set_title(f"{title} (RGB)")
    elif img_np.ndim == 2 or (img_np.ndim == 3 and img_np.shape[2] == 1): # Grayscale
        if img_np.ndim == 3: # Squeeze single channel if needed
             img_np = img_np.squeeze(axis=2)
        hist = cv2.calcHist([img_np], [0], None, [256], [0, 256])
        ax.plot(hist, color='black', label='Intensity')
        ax.legend()
        ax.set_title(f"{title} (Grayscale)")
    else:
        warnings.warn(f"Cannot plot histogram: Unsupported image shape {img_np.shape}")
        plt.close(fig) # Close the empty figure
        return None

    ax.set_xlabel("Pixel Value")
    ax.set_ylabel("Frequency")
    ax.set_xlim([0, 256])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        try:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path)
            print(f"Histogram saved to: {save_path}")
        except Exception as e:
            warnings.warn(f"Failed to save histogram to {save_path}: {e}")
            
    return fig

# --- Deprecated / Renamed --- 
# compare_original_result is replaced by compare_images
# show_processing_steps is replaced by compare_images or display_image_grid

# def compare_original_result(original, mask, result):
#     """
#     Show original, mask, and result side by side.
#     DEPRECATED: Use compare_images instead.
#     """
#     warnings.warn("`compare_original_result` is deprecated. Use `compare_images`.", DeprecationWarning)
#     return compare_images([original, mask, result], titles=['Original', 'Mask', 'Result'])

# def show_processing_steps(images, titles):
#     """
#     Show the step-by-step processing of an image.
#     DEPRECATED: Use compare_images or display_image_grid.
#     """
#     warnings.warn("`show_processing_steps` is deprecated. Use `compare_images` or `display_image_grid`.", DeprecationWarning)
#     return compare_images(images, titles=titles) 