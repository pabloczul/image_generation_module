"""
Image Mask Filtering and Refinement Utilities.
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple
import logging # Added

# --- Morphological Refinement --- 

def refine_morphological(mask: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    """
    Refines a mask using morphological operations defined in the config dictionary.
    (Moved from Segmenter class)

    Args:
        mask (np.ndarray): Grayscale alpha mask (0-255) or binary mask.
        config (Dict[str, Any]): Configuration dictionary containing keys like
                                 'mask_opening_kernel_size', 'mask_opening_iterations',
                                 'mask_closing_kernel_size', 'mask_closing_iterations',
                                 'mask_dilation_kernel_size', 'mask_dilation_iterations'.

    Returns:
        np.ndarray: The refined mask (0-255, uint8).
    """
    if not isinstance(mask, np.ndarray):
        raise TypeError(f"Input mask must be a NumPy array, got {type(mask)}")

    refined_mask = mask.astype(np.uint8)
    if refined_mask.ndim == 3 and refined_mask.shape[2] == 1:
        refined_mask = refined_mask.squeeze()
    elif refined_mask.ndim != 2:
        raise ValueError(f"Input mask must be a single channel (grayscale or binary), shape was {mask.shape}")

    # --- Morphological Opening (Remove noise) ---
    open_k_size = config.get('mask_opening_kernel_size', 0)
    open_iter = config.get('mask_opening_iterations', 1)
    if open_k_size > 0 and open_k_size % 2 != 0: # Kernel size must be positive odd
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (open_k_size, open_k_size))
        refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel_open, iterations=open_iter)
        logging.debug(f"Applied morphological opening: k_size={open_k_size}, iter={open_iter}")
    elif open_k_size > 0:
        logging.warning(f"mask_opening_kernel_size ({open_k_size}) must be odd, skipping opening.")

    # --- Morphological Closing (Fill holes) ---
    close_k_size = config.get('mask_closing_kernel_size', 0)
    close_iter = config.get('mask_closing_iterations', 1)
    if close_k_size > 0 and close_k_size % 2 != 0: # Kernel size must be positive odd
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (close_k_size, close_k_size))
        refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel_close, iterations=close_iter)
        logging.debug(f"Applied morphological closing: k_size={close_k_size}, iter={close_iter}")
    elif close_k_size > 0:
        logging.warning(f"mask_closing_kernel_size ({close_k_size}) must be odd, skipping closing.")

    # --- Morphological Dilation (Expand mask slightly) ---
    dilate_k_size = config.get('mask_dilation_kernel_size', 0)
    dilate_iter = config.get('mask_dilation_iterations', 1)
    if dilate_k_size > 0 and dilate_k_size % 2 != 0: # Kernel size must be positive odd
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_k_size, dilate_k_size))
        refined_mask = cv2.dilate(refined_mask, kernel_dilate, iterations=dilate_iter)
        logging.debug(f"Applied morphological dilation: k_size={dilate_k_size}, iter={dilate_iter}")
    elif dilate_k_size > 0:
        logging.warning(f"mask_dilation_kernel_size ({dilate_k_size}) must be odd, skipping dilation.")

    return refined_mask

# --- Placeholder Filter Functions --- 
# These will be implemented in subsequent steps.

def filter_by_contrast(image_np: np.ndarray, mask_np: np.ndarray, config: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Checks if the contrast between the masked object and its immediate background is sufficient.
    Rejects the mask if the contrast is too low.

    Args:
        image_np (np.ndarray): The original input image (BGR or RGB format).
        mask_np (np.ndarray): The binary segmentation mask (0 or 255, uint8).
        config (Dict[str, Any]): Configuration dictionary containing:
            'contrast_filter_threshold' (float): Minimum absolute mean difference in L channel.
            'contrast_band_width' (int): Pixel width for inner/outer bands.

    Returns:
        Tuple[bool, str]: (True, "Contrast sufficient") if contrast is high enough,
                          (False, "Reason for rejection") otherwise.
    """
    threshold = config.get('contrast_filter_threshold', 10.0)
    band_width = config.get('contrast_band_width', 3)

    if band_width <= 0:
        logging.warning("Contrast band width must be positive, skipping filter.")
        return True, "Invalid band width"

    # Ensure mask is binary 0/255
    if mask_np.dtype != np.uint8 or set(np.unique(mask_np)) - {0, 255}:
        _, binary_mask = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)
    else:
        binary_mask = mask_np

    if np.count_nonzero(binary_mask) == 0:
        logging.debug("Skipping contrast filter: Empty mask.")
        return True, "Empty mask" # Pass empty masks

    try:
        # Create kernel for erosion/dilation based on band width
        # Kernel size should be odd: 2 * width + 1
        k_size = 2 * band_width + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))

        # Erode mask to get inner region for band calculation
        inner_mask = cv2.erode(binary_mask, kernel, iterations=1)
        # Dilate mask to get outer region for band calculation
        outer_boundary = cv2.dilate(binary_mask, kernel, iterations=1)

        # Define bands
        inner_band = cv2.subtract(binary_mask, inner_mask)
        outer_band = cv2.subtract(outer_boundary, binary_mask)

        # Convert image to LAB color space for luminance (L channel)
        # Check if image is BGR or RGB first (common issue)
        if image_np.ndim == 3 and image_np.shape[2] == 3:
             # Heuristic: Assume BGR if not explicitly known, as OpenCV reads BGR
             # If your pipeline guarantees RGB, change this to COLOR_RGB2LAB
             lab_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2LAB)
        else: # Grayscale or unexpected format
             logging.warning(f"Contrast filter expects BGR/RGB image, got shape {image_np.shape}. Using grayscale.")
             if image_np.ndim == 3:
                 gray_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY) # Assume BGR source
             else:
                 gray_image = image_np
             lab_image = cv2.cvtColor(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2LAB)

        l_channel = lab_image[:, :, 0]

        # Calculate mean L value in inner and outer bands
        inner_pixels = l_channel[inner_band == 255]
        outer_pixels = l_channel[outer_band == 255]

        if inner_pixels.size == 0 or outer_pixels.size == 0:
            logging.warning("Contrast filter: Could not define inner or outer band (mask too small or thin?). Passing.")
            return True, "Could not define contrast bands"

        mean_inner_l = np.mean(inner_pixels)
        mean_outer_l = np.mean(outer_pixels)

        contrast_diff = abs(mean_inner_l - mean_outer_l)
        logging.debug(f"Contrast check: Mean Inner L={mean_inner_l:.2f}, Mean Outer L={mean_outer_l:.2f}, Diff={contrast_diff:.2f}, Threshold={threshold}")

        if contrast_diff < threshold:
            reason = f"Low contrast: mean L diff {contrast_diff:.2f} < threshold {threshold:.2f}"
            logging.info(f"Mask rejected by contrast filter: {reason}")
            return False, reason
        else:
            return True, "Contrast sufficient"

    except Exception as e:
        logging.error(f"Error during contrast filtering: {e}", exc_info=True)
        return False, f"Error during contrast filter: {e}"

def filter_by_text_overlap(image_np: np.ndarray, mask_np: np.ndarray, config: Dict[str, Any]) -> Tuple[bool, str]:
    """Placeholder for text overlap filter. (Implementation Skipped)"""
    logging.debug("Skipping text filter (implementation skipped).")
    return True, "Text filter skipped"

def filter_by_clutter(image_np: np.ndarray, mask_np: np.ndarray, config: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Filters masks based on object detection results to handle clutter.
    - Detects objects in the image using a pre-trained model (e.g., YOLO).
    - Identifies the primary object (heuristic: largest area).
    - Checks if the mask aligns well with the primary object's bounding box.
    - Checks if the mask significantly overlaps with other detected objects.

    Requires `ultralytics` library: pip install ultralytics

    Args:
        image_np (np.ndarray): Original input image (BGR or RGB format).
        mask_np (np.ndarray): Binary segmentation mask (0 or 255, uint8).
        config (Dict[str, Any]): Configuration dictionary containing:
            'clutter_detector_model' (str): Name/path of the YOLO model (e.g., 'yolov8n.pt').
            'clutter_min_primary_iou' (float): Min IoU between mask and primary object box.
            'clutter_max_other_overlap' (float): Max overlap ratio allowed between mask and non-primary objects.
            'clutter_confidence_threshold' (float): Confidence threshold for YOLO detections.

    Returns:
        Tuple[bool, str]: (True, "Clutter check passed") or (False, "Reason for rejection").
    """
    model_name = config.get('clutter_detector_model', 'yolov8n.pt')
    min_iou_primary = config.get('clutter_min_primary_iou', 0.4)
    max_overlap_other = config.get('clutter_max_other_overlap', 0.3)
    conf_threshold = config.get('clutter_confidence_threshold', 0.25) # YOLO default

    # --- Dependency Check --- 
    try:
        from ultralytics import YOLO
        # TODO: Consider pre-loading the model in the pipeline __init__ for efficiency
        model = YOLO(model_name)
    except ImportError:
        logging.error("Clutter filter requires 'ultralytics'. Install with: pip install ultralytics")
        return False, "ultralytics library not installed"
    except Exception as model_load_e:
        logging.error(f"Failed to load YOLO model '{model_name}': {model_load_e}")
        return False, f"Failed to load detector model {model_name}"
    # ------------------------

    # Ensure mask is binary 0/255
    if mask_np.dtype != np.uint8 or set(np.unique(mask_np)) - {0, 255}:
        _, binary_mask = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)
    else:
        binary_mask = mask_np

    mask_area = np.count_nonzero(binary_mask)
    if mask_area == 0:
        logging.debug("Skipping clutter filter: Empty mask.")
        return True, "Empty mask" # Pass empty masks

    try:
        # Run YOLO detection
        # Assuming image_np is BGR or RGB, YOLO should handle it.
        results = model.predict(source=image_np, conf=conf_threshold, verbose=False) # verbose=False to reduce console spam
        
        if not results or len(results) == 0 or not results[0].boxes:
            logging.warning("Clutter filter: No objects detected by YOLO. Passing mask.")
            return True, "No objects detected"

        # Get bounding boxes [x1, y1, x2, y2] and areas
        boxes = results[0].boxes.xyxy.cpu().numpy() # [n, 4]
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) # [n]

        if len(areas) == 0:
             logging.warning("Clutter filter: No objects detected after processing results. Passing mask.")
             return True, "No objects detected"
             
        # --- Identify Primary Object (Heuristic: Largest Area) ---
        primary_idx = np.argmax(areas)
        primary_box = boxes[primary_idx]
        primary_box_area = areas[primary_idx]
        logging.debug(f"Clutter filter: Primary object box (largest area): {primary_box.astype(int)}")
        # --------------------------------------------------------

        # --- Calculate IoU between Mask and Primary Object Box ---
        # Get mask bounding box
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
             logging.warning("Clutter filter: Could not find contours for mask. Passing.")
             return True, "Mask has no contours"
        
        # Combine contours if multiple exist (e.g., fragmented mask)
        all_points = np.concatenate(contours, axis=0)
        mask_x, mask_y, mask_w, mask_h = cv2.boundingRect(all_points)
        mask_box = [mask_x, mask_y, mask_x + mask_w, mask_y + mask_h]

        # Intersection coordinates
        x_left = max(mask_box[0], primary_box[0])
        y_top = max(mask_box[1], primary_box[1])
        x_right = min(mask_box[2], primary_box[2])
        y_bottom = min(mask_box[3], primary_box[3])

        intersection_area = 0
        if x_right > x_left and y_bottom > y_top:
            intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # Union area = Area(box1) + Area(box2) - IntersectionArea
        # Use mask_area calculated earlier for more accuracy than mask bounding box area
        union_area = mask_area + primary_box_area - intersection_area
        
        iou = intersection_area / union_area if union_area > 0 else 0
        logging.debug(f"Clutter filter: Mask Area={mask_area:.0f}, Primary Box Area={primary_box_area:.0f}, Intersection={intersection_area:.0f}, Union={union_area:.0f}, IoU={iou:.3f}")

        if iou < min_iou_primary:
            reason = f"Low IoU with primary object ({iou:.3f} < {min_iou_primary:.3f})"
            logging.info(f"Mask rejected by clutter filter: {reason}")
            return False, reason
        # -----------------------------------------------------------

        # --- Check Overlap with Other Objects ---
        mask_overlap_ratio_total = 0
        if len(boxes) > 1:
            mask_canvas = np.zeros(binary_mask.shape[:2], dtype=np.uint8)
            mask_canvas = cv2.rectangle(mask_canvas, (mask_box[0], mask_box[1]), (mask_box[2], mask_box[3]), 255, -1)
            
            total_other_intersection = 0
            for i, box in enumerate(boxes):
                if i == primary_idx: continue # Skip primary object

                # Intersection with this non-primary box
                other_x_left = max(mask_box[0], box[0])
                other_y_top = max(mask_box[1], box[1])
                other_x_right = min(mask_box[2], box[2])
                other_y_bottom = min(mask_box[3], box[3])

                if other_x_right > other_x_left and other_y_bottom > other_y_top:
                    # More precise: calculate intersection of mask pixels with box area
                    other_box_mask = np.zeros_like(binary_mask)
                    cv2.rectangle(other_box_mask, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), 255, -1)
                    intersection_with_other = np.count_nonzero(cv2.bitwise_and(binary_mask, other_box_mask))
                    total_other_intersection += intersection_with_other
                    # logging.debug(f"Overlap with non-primary obj {i}: {intersection_with_other} pixels")
            
            mask_overlap_ratio_total = total_other_intersection / mask_area if mask_area > 0 else 0
            logging.debug(f"Clutter filter: Total intersection with non-primary objects: {total_other_intersection} pixels, Mask area: {mask_area}, Overlap Ratio: {mask_overlap_ratio_total:.3f}")
            
            if mask_overlap_ratio_total > max_overlap_other:
                 reason = f"High overlap with other objects ({mask_overlap_ratio_total:.3f} > {max_overlap_other:.3f})"
                 logging.info(f"Mask rejected by clutter filter: {reason}")
                 return False, reason
        # -------------------------------------

        return True, "Clutter check passed"

    except Exception as e:
        logging.error(f"Error during clutter filtering: {e}", exc_info=True)
        return False, f"Error during clutter filter: {e}"

def filter_by_contour(image_np: np.ndarray, mask_np: np.ndarray, config: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Filters masks based on geometric properties of their contours.
    Checks for excessive complexity, fragmentation, or unreasonable size.

    Args:
        image_np (np.ndarray): Original input image (used for size context).
        mask_np (np.ndarray): Binary segmentation mask (0 or 255, uint8).
        config (Dict[str, Any]): Configuration dictionary containing:
            'contour_max_points' (int): Max allowed points in the largest contour.
            'contour_max_count' (int): Max number of distinct contours allowed.
            'contour_min_solidity' (float): Min solidity (area / convex hull area).
            'contour_min_area_ratio' (float): Min ratio of mask area to image area.
            'contour_max_area_ratio' (float): Max ratio of mask area to image area.

    Returns:
        Tuple[bool, str]: (True, "Contour properties acceptable") or (False, "Reason for rejection").
    """
    max_count = config.get('contour_max_count', 5)
    max_points = config.get('contour_max_points', 2500)
    min_solidity = config.get('contour_min_solidity', 0.70)
    min_area_ratio = config.get('contour_min_area_ratio', 0.01) # 1% of image area
    max_area_ratio = config.get('contour_max_area_ratio', 0.95) # 95% of image area

    # Ensure mask is binary 0/255
    if mask_np.dtype != np.uint8 or set(np.unique(mask_np)) - {0, 255}:
        _, binary_mask = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)
    else:
        binary_mask = mask_np

    mask_area = np.count_nonzero(binary_mask)
    image_area = image_np.shape[0] * image_np.shape[1]
    if image_area == 0:
        return False, "Invalid image area (zero)"
        
    if mask_area == 0:
        logging.debug("Skipping contour filter: Empty mask.")
        return True, "Empty mask" # Pass empty masks

    try:
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        num_contours = len(contours)

        # --- Check Contour Count --- 
        if num_contours == 0:
             logging.warning("Contour filter: Could not find contours. Passing.")
             return True, "Mask has no contours"
        if num_contours > max_count:
            reason = f"Too many contours ({num_contours} > {max_count})"
            logging.info(f"Mask rejected by contour filter: {reason}")
            return False, reason
        # -------------------------

        # --- Check Area Ratio --- 
        area_ratio = mask_area / image_area
        if not (min_area_ratio <= area_ratio <= max_area_ratio):
            reason = f"Mask area ratio out of bounds ({area_ratio:.3f}, expected [{min_area_ratio:.3f}-{max_area_ratio:.3f}])"
            logging.info(f"Mask rejected by contour filter: {reason}")
            return False, reason
        # ------------------------
        
        # --- Analyze Largest Contour --- 
        largest_contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(largest_contour)
        if contour_area <= 0:
             logging.warning("Contour filter: Largest contour area is zero. Passing.")
             return True, "Zero area largest contour"

        # Check point count (complexity)
        num_points = len(largest_contour)
        if num_points > max_points:
            reason = f"Largest contour too complex ({num_points} points > {max_points})"
            logging.info(f"Mask rejected by contour filter: {reason}")
            return False, reason
        
        # Check solidity (convexity)
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        solidity = contour_area / hull_area if hull_area > 0 else 0
        if solidity < min_solidity:
            reason = f"Low solidity ({solidity:.3f} < {min_solidity:.3f})"
            logging.info(f"Mask rejected by contour filter: {reason}")
            return False, reason
        # -----------------------------
        
        logging.debug(f"Contour filter checks passed: Count={num_contours}, AreaRatio={area_ratio:.3f}, Points={num_points}, Solidity={solidity:.3f}")
        return True, "Contour properties acceptable"

    except Exception as e:
        logging.error(f"Error during contour filtering: {e}", exc_info=True)
        return False, f"Error during contour filter: {e}" 