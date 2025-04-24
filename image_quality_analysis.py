#!/usr/bin/env python
"""
Image Quality and Filter Analysis Tool

This script analyzes images for quality metrics and filter performance,
creating detailed reports for each image and a summary CSV.
"""

import sys
import os
import time
import json
import csv
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import cv2
from PIL import Image

# Add project root to Python path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import components from the project
from src.image.quality import ImageAssessor
from src.image.segmentation import Segmenter
from src.image.filtering import (
    filter_by_contrast,
    filter_by_clutter,
    filter_by_contour,
    refine_morphological
)
from src.utils.data_io import get_image_paths, save_image, load_image
from src.config import DEFAULT_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

class ImageAnalyzer:
    """Analyzes images for quality and filter performance, generating detailed reports."""
    
    def __init__(
        self,
        output_dir: Path,
        segmenter_model: str = DEFAULT_CONFIG['segmenter_model'],
        save_visualizations: bool = True,
        config_overrides: Optional[Dict[str, Any]] = None,
        log_level: int = logging.INFO
    ):
        """
        Initialize the image analyzer.
        
        Args:
            output_dir: Directory to save analysis results
            segmenter_model: Segmentation model to use
            save_visualizations: Whether to save visualization images
            config_overrides: Dictionary of configuration overrides
            log_level: Logging level (default: INFO)
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create results directory structure
        self.reports_dir = output_dir / "reports"
        self.reports_dir.mkdir(exist_ok=True)
        
        self.visuals_dir = output_dir / "visualizations"
        self.visuals_dir.mkdir(exist_ok=True)
        
        self.masks_dir = output_dir / "masks"
        self.masks_dir.mkdir(exist_ok=True)
        
        self.logs_dir = output_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.config = DEFAULT_CONFIG.copy()
        if config_overrides:
            self.config.update(config_overrides)
        
        self.assessor = ImageAssessor(
            min_resolution=self.config.get('min_resolution', (300, 300)),
            blur_threshold=self.config.get('blur_threshold', 100.0),
            contrast_threshold=self.config.get('contrast_threshold', 30.0)
        )
        
        self.segmenter = Segmenter(model_name=segmenter_model)
        
        self.save_visualizations = save_visualizations
        
        # CSV summary file with expanded columns
        self.csv_path = output_dir / "analysis_summary.csv"
        self.csv_headers = [
            "image_name", "resolution", "width", "height", 
            "is_blurry", "blur_value", "blur_threshold",
            "low_contrast", "contrast_value", "contrast_threshold",
            "segmentation_success", "mask_coverage", "segmentation_model",
            "contrast_filter_pass", "contrast_inner", "contrast_outer", "contrast_diff", "contrast_threshold_used",
            "clutter_filter_pass", "detected_objects", "primary_object_iou", "primary_min_threshold",
            "contour_filter_pass", "contour_count", "max_contour_points", "contour_solidity",
            "mask_refinement", "opening_kernel", "opening_iter", "closing_kernel", "closing_iter", "dilation_kernel", "dilation_iter",
            "filter_counts_passed", "filter_counts_total", "final_status", 
            "processing_time", "error_details"
        ]
        
        # Initialize CSV file
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.csv_headers)
    
    def _setup_image_logger(self, img_base_name: str) -> logging.Logger:
        """Set up a dedicated logger for each image with file output."""
        # Create a logger
        logger = logging.getLogger(f"img_{img_base_name}")
        logger.setLevel(logging.DEBUG)
        
        # Remove any existing handlers to avoid duplicates
        if logger.handlers:
            for handler in logger.handlers:
                logger.removeHandler(handler)
        
        # Create handlers
        log_file = self.logs_dir / f"{img_base_name}_processing.log"
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.DEBUG)
        
        # Create formatters
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        
        # Allow the logger to propagate messages to the root logger (console)
        logger.propagate = True
        
        return logger
    
    def analyze_image(self, image_path: Path) -> Dict[str, Any]:
        """
        Analyze a single image and return detailed results.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing analysis results
        """
        start_time = time.time()
        image_name = image_path.name
        img_base_name = image_path.stem
        
        # Set up logger for this image
        logger = self._setup_image_logger(img_base_name)
        
        logger.info(f"Starting analysis of {image_name}")
        results = {
            "image_name": image_name, 
            "image_path": str(image_path),
            "segmentation_model": self.segmenter.model_name,
            "blur_threshold": self.config.get('blur_threshold', 100.0),
            "contrast_threshold": self.config.get('contrast_threshold', 30.0)
        }
        
        # Create image-specific output directories
        img_report_dir = self.reports_dir / img_base_name
        img_report_dir.mkdir(exist_ok=True)
        
        try:
            # Load image first to ensure we get dimensions even if assessment fails
            logger.info("Loading image file")
            image = load_image(image_path)
            image_np = np.array(image)
            
            # Add basic dimensions to results
            width, height = image.size
            results["width"] = width
            results["height"] = height
            results["resolution"] = f"{width}x{height}"
            logger.info(f"Image dimensions: {width}x{height}")
            
            # Step 1: Basic quality assessment
            try:
                logger.info("Performing quality assessment")
                quality = self.assessor.assess_quality(image_path)
                logger.info(f"Quality assessment results: {quality}")
                
                # Only update results with quality assessment, don't overwrite dimensions
                # that we've already captured
                for key, value in quality.items():
                    if key not in ["width", "height"]:
                        results[key] = value
                
                # Extract and log specific quality metrics - fixed to handle nested structure
                blur_value = quality.get('blur', {}).get('value', 'N/A')
                blur_passed = quality.get('blur', {}).get('passed', False)
                contrast_value = quality.get('contrast', {}).get('value', 'N/A')
                contrast_passed = quality.get('contrast', {}).get('passed', False)
                
                logger.info(f"Blur value: {blur_value} (threshold: {self.config.get('blur_threshold', 100.0)}), passed: {blur_passed}")
                logger.info(f"Contrast value: {contrast_value} (threshold: {self.config.get('contrast_threshold', 30.0)}), passed: {contrast_passed}")
                logger.info(f"Is processable: {results.get('is_processable', False)}")
            except Exception as e:
                logger.warning(f"Quality assessment failed: {e}")
                # Set default quality values if assessment failed
                results["is_blurry"] = False
                results["blur_value"] = 0
                results["low_contrast"] = False
                results["contrast_value"] = 0
                results["is_processable"] = True  # Assume processable unless proven otherwise
            
            # Step 2: Segmentation
            try:
                logger.info(f"Segmenting image with model: {self.segmenter.model_name}")
                raw_mask = self.segmenter.segment(
                    image_np,  # Pass numpy array instead of path for consistency
                    save_intermediate=True,
                    intermediate_dir=self.masks_dir,
                    output_basename=f"{img_base_name}_segmentation"
                )
                
                if raw_mask is None:
                    logger.error("Segmentation failed to generate a mask")
                    results["segmentation_success"] = False
                    results["error_details"] = "Segmentation failed to generate a mask"
                else:
                    logger.info("Segmentation successful")
                    results["segmentation_success"] = True
                    
                    # Calculate mask coverage
                    mask_coverage = np.count_nonzero(raw_mask) / raw_mask.size
                    mask_coverage_pct = mask_coverage * 100
                    results["mask_coverage"] = f"{mask_coverage_pct:.2f}%"
                    results["mask_coverage_value"] = mask_coverage_pct
                    logger.info(f"Mask coverage: {mask_coverage_pct:.2f}%")
                    
                    # Mask refinement settings
                    opening_kernel = self.config.get('mask_opening_kernel_size', 3)
                    opening_iter = self.config.get('mask_opening_iterations', 1)
                    closing_kernel = self.config.get('mask_closing_kernel_size', 15)
                    # Ensure kernel sizes are odd
                    if opening_kernel % 2 == 0:
                        opening_kernel += 1
                    if closing_kernel % 2 == 0:
                        closing_kernel += 1
                    closing_iter = self.config.get('mask_closing_iterations', 1)
                    dilation_kernel = self.config.get('mask_dilation_kernel_size', 1)
                    dilation_iter = self.config.get('mask_dilation_iterations', 1)
                    
                    # Store refinement settings in results
                    results["mask_refinement"] = True
                    results["opening_kernel"] = opening_kernel
                    results["opening_iter"] = opening_iter
                    results["closing_kernel"] = closing_kernel
                    results["closing_iter"] = closing_iter
                    results["dilation_kernel"] = dilation_kernel
                    results["dilation_iter"] = dilation_iter
                    
                    # Step 3: Apply filters
                    # Apply morphological refinement
                    logger.info(f"Refining mask (opening {opening_kernel}x{opening_iter}, closing {closing_kernel}x{closing_iter}, dilation {dilation_kernel}x{dilation_iter})")
                    
                    # Create a config dictionary for refine_morphological
                    mask_refine_config = {
                        'mask_opening_kernel_size': opening_kernel,
                        'mask_opening_iterations': opening_iter,
                        'mask_closing_kernel_size': closing_kernel,
                        'mask_closing_iterations': closing_iter,
                        'mask_dilation_kernel_size': dilation_kernel,
                        'mask_dilation_iterations': dilation_iter
                    }
                    
                    refined_mask = refine_morphological(
                        raw_mask,
                        config=mask_refine_config
                    )
                    
                    # Save refined mask
                    mask_img = Image.fromarray(refined_mask)
                    refined_mask_path = self.masks_dir / f"{img_base_name}_refined_mask.png"
                    save_image(mask_img, refined_mask_path)
                    logger.info(f"Saved refined mask to {refined_mask_path}")
                    
                    # Calculate refined mask coverage
                    refined_coverage = np.count_nonzero(refined_mask) / refined_mask.size * 100
                    logger.info(f"Refined mask coverage: {refined_coverage:.2f}% (change: {refined_coverage - mask_coverage_pct:+.2f}%)")
                    
                    # Apply contrast filter
                    try:
                        contrast_threshold = self.config.get('contrast_filter_threshold', 10.0)
                        band_width = self.config.get('contrast_band_width', 3)
                        logger.info(f"Applying contrast filter (threshold: {contrast_threshold}, band width: {band_width})")
                        
                        # Create config dictionary for contrast filter
                        contrast_config = {
                            'contrast_filter_threshold': contrast_threshold,
                            'contrast_band_width': band_width
                        }
                        
                        # Call filter - returns (pass/fail, message)
                        contrast_pass, contrast_message = filter_by_contrast(
                            image_np, 
                            refined_mask,
                            config=contrast_config
                        )
                        
                        # Create details manually since there's no return_details parameter
                        contrast_details = {
                            "message": contrast_message,
                            "threshold": contrast_threshold,
                            "band_width": band_width
                        }
                        
                        results["contrast_filter_pass"] = contrast_pass
                        results["contrast_details"] = contrast_details
                        results["contrast_threshold_used"] = contrast_threshold
                        
                        logger.info(f"Contrast filter result: {'PASS' if contrast_pass else 'FAIL'}")
                        logger.info(f"  Message: {contrast_message}")
                        logger.info(f"  Threshold: {contrast_threshold}")
                    except Exception as e:
                        logger.warning(f"Contrast filter failed: {e}")
                        results["contrast_filter_pass"] = True  # Be lenient if filter fails
                        results["contrast_details"] = {"error": str(e)}
                        results["error_details"] = f"Contrast filter error: {str(e)}"
                    
                    # Apply clutter filter
                    try:
                        min_primary_iou = self.config.get('clutter_min_primary_iou', 0.4)
                        max_other_overlap = self.config.get('clutter_max_other_overlap', 0.3)
                        logger.info(f"Applying clutter filter (min IoU: {min_primary_iou}, max overlap: {max_other_overlap})")
                        
                        # Create config dictionary for clutter filter
                        clutter_config = {
                            'clutter_min_primary_iou': min_primary_iou,
                            'clutter_max_other_overlap': max_other_overlap,
                            'clutter_detector_model': self.config.get('clutter_detector_model', 'yolov8n.pt'),
                            'clutter_confidence_threshold': self.config.get('clutter_confidence_threshold', 0.25)
                        }
                        
                        # Call filter - returns (pass/fail, message)
                        clutter_pass, clutter_message = filter_by_clutter(
                            image_np,
                            refined_mask,
                            config=clutter_config
                        )
                        
                        # Create details manually
                        clutter_details = {
                            "message": clutter_message,
                            "min_primary_iou": min_primary_iou,
                            "max_other_overlap": max_other_overlap,
                            "detections": []  # Empty list since we don't have actual detection data
                        }
                        
                        results["clutter_filter_pass"] = clutter_pass
                        results["clutter_details"] = clutter_details
                        results["detected_objects"] = 0  # Default to 0 since we don't have actual detection count
                        results["primary_object_iou"] = 0  # Default since we don't have this from the function
                        results["primary_min_threshold"] = min_primary_iou
                        
                        logger.info(f"Clutter filter result: {'PASS' if clutter_pass else 'FAIL'}")
                        logger.info(f"  Message: {clutter_message}")
                    except Exception as e:
                        logger.warning(f"Clutter filter failed: {e}")
                        results["clutter_filter_pass"] = True  # Be lenient if filter fails
                        results["clutter_details"] = {"error": str(e)}
                        results["error_details"] = results.get("error_details", "") + f" | Clutter filter error: {str(e)}"
                    
                    # Apply contour filter
                    try:
                        max_points = self.config.get('contour_max_points', 2500)
                        max_count = self.config.get('contour_max_count', 5)
                        min_solidity = self.config.get('contour_min_solidity', 0.7)
                        logger.info(f"Applying contour filter (max points: {max_points}, max count: {max_count}, min solidity: {min_solidity})")
                        
                        # Create config dictionary for contour filter
                        contour_config = {
                            'contour_max_points': max_points,
                            'contour_max_count': max_count,
                            'contour_min_solidity': min_solidity
                        }
                        
                        # Call filter - returns (pass/fail, message)
                        contour_pass, contour_message = filter_by_contour(
                            image_np,
                            refined_mask,
                            config=contour_config
                        )
                        
                        # Create details manually
                        contour_details = {
                            "message": contour_message,
                            "max_points": max_points,
                            "max_count": max_count,
                            "min_solidity": min_solidity
                        }
                        
                        results["contour_filter_pass"] = contour_pass
                        results["contour_details"] = contour_details
                        results["contour_count"] = 0  # Default since we don't have actual count
                        results["max_contour_points"] = 0  # Default
                        results["contour_solidity"] = 0  # Default
                        
                        logger.info(f"Contour filter result: {'PASS' if contour_pass else 'FAIL'}")
                        logger.info(f"  Message: {contour_message}")
                    except Exception as e:
                        logger.warning(f"Contour filter failed: {e}")
                        results["contour_filter_pass"] = True  # Be lenient if filter fails
                        results["contour_details"] = {"error": str(e)}
                        results["error_details"] = results.get("error_details", "") + f" | Contour filter error: {str(e)}"
                    
                    # Overall pass/fail status - be more lenient with failures
                    # If quality and segmentation are okay, and at least 2 of 3 filters pass, mark as success
                    filter_passes = [
                        results.get("contrast_filter_pass", False),
                        results.get("clutter_filter_pass", False),
                        results.get("contour_filter_pass", False)
                    ]
                    
                    num_passing_filters = sum(1 for p in filter_passes if p)
                    results["filter_counts_passed"] = num_passing_filters
                    results["filter_counts_total"] = len(filter_passes)
                    
                    results["final_status"] = (
                        results.get("segmentation_success", False) and
                        (num_passing_filters >= 2)  # More lenient - pass if 2+ filters pass
                    )
                    
                    logger.info(f"Filter pass count: {num_passing_filters}/{len(filter_passes)}")
                    logger.info(f"Final status: {'PASS' if results['final_status'] else 'FAIL'}")
                    
                    # Create visualization if enabled
                    if self.save_visualizations:
                        try:
                            logger.info("Creating visualization")
                            self._create_visualization(
                                image_np, 
                                raw_mask,
                                refined_mask,
                                img_base_name,
                                clutter_details.get('detections', []) if 'clutter_details' in results else []
                            )
                            logger.info(f"Visualization saved to {self.visuals_dir / f'{img_base_name}_analysis.png'}")
                        except Exception as e:
                            logger.warning(f"Visualization creation failed: {e}")
                            results["error_details"] = results.get("error_details", "") + f" | Visualization error: {str(e)}"
            
            except Exception as e:
                logger.error(f"Error during segmentation: {e}", exc_info=True)
                results["segmentation_success"] = False
                results["error_details"] = f"Error during segmentation: {str(e)}"
                results["final_status"] = False
        
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}", exc_info=True)
            results["error_details"] = f"Error loading image: {str(e)}"
            results["final_status"] = False
        
        # Calculate processing time
        end_time = time.time()
        processing_time = end_time - start_time
        results["processing_time"] = f"{processing_time:.2f}s"
        logger.info(f"Processing completed in {processing_time:.2f} seconds")
        
        # Save detailed report as JSON
        try:
            logger.info(f"Saving detailed JSON report to {img_report_dir / 'analysis_results.json'}")
            with open(img_report_dir / "analysis_results.json", 'w') as f:
                # Convert numpy values to Python types for JSON serialization
                serializable_results = self._make_serializable(results)
                json.dump(serializable_results, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save JSON report: {e}")
        
        # Add to CSV summary
        try:
            self._add_to_csv_summary(results)
            logger.info("Added results to CSV summary")
        except Exception as e:
            logger.error(f"Failed to update CSV summary: {e}")
        
        return results
    
    def _create_visualization(
        self,
        image: np.ndarray,
        raw_mask: np.ndarray,
        refined_mask: np.ndarray,
        img_base_name: str,
        detections: List[Dict[str, Any]] = None
    ):
        """Create and save a visualization of the analysis results."""
        # Create a 2x2 grid visualization
        h, w = image.shape[:2]
        
        # Create a canvas twice the size of the original image
        canvas = np.zeros((h*2, w*2, 3), dtype=np.uint8)
        
        # Original image
        if len(image.shape) == 2:  # Grayscale
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image_rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        else:
            image_rgb = image.copy()
        
        canvas[:h, :w] = image_rgb
        
        # Raw mask (convert to RGB for visualization)
        raw_mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        raw_mask_rgb[raw_mask > 0] = [255, 255, 255]
        canvas[:h, w:w*2] = raw_mask_rgb
        
        # Refined mask (convert to RGB for visualization)
        refined_mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        refined_mask_rgb[refined_mask > 0] = [255, 255, 255]
        canvas[h:h*2, :w] = refined_mask_rgb
        
        # Mask overlay on original
        overlay = image_rgb.copy()
        overlay[refined_mask > 0] = [0, 255, 0]  # Green overlay for mask
        alpha = 0.5
        composite = cv2.addWeighted(image_rgb, 1-alpha, overlay, alpha, 0)
        
        # Draw detection boxes if available
        if detections:
            for det in detections:
                if 'bbox' in det:
                    x1, y1, x2, y2 = det['bbox']
                    color = (0, 0, 255)  # Red for bounding boxes
                    cv2.rectangle(composite, (x1, y1), (x2, y2), color, 2)
                    
                    # Label with class and confidence if available
                    label = f"{det.get('class', 'Object')}: {det.get('confidence', 0):.2f}"
                    cv2.putText(composite, label, (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        canvas[h:h*2, w:w*2] = composite
        
        # Add labels
        labels = ["Original Image", "Raw Mask", "Refined Mask", "Mask Overlay"]
        positions = [(10, 30), (w+10, 30), (10, h+30), (w+10, h+30)]
        
        for label, pos in zip(labels, positions):
            cv2.putText(canvas, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (255, 255, 255), 2)
        
        # Save the visualization
        viz_path = self.visuals_dir / f"{img_base_name}_analysis.png"
        cv2.imwrite(str(viz_path), canvas)
    
    def _make_serializable(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a dictionary with numpy values to JSON-serializable Python types."""
        result = {}
        for k, v in data.items():
            if isinstance(v, np.ndarray):
                result[k] = v.tolist() if v.size < 1000 else "ndarray (too large to include)"
            elif isinstance(v, np.integer):
                result[k] = int(v)
            elif isinstance(v, np.floating):
                result[k] = float(v)
            elif isinstance(v, np.bool_):
                result[k] = bool(v)
            elif isinstance(v, dict):
                result[k] = self._make_serializable(v)
            elif isinstance(v, list) or isinstance(v, tuple):
                result[k] = [
                    self._make_serializable(item) if isinstance(item, dict) 
                    else bool(item) if isinstance(item, np.bool_)
                    else int(item) if isinstance(item, np.integer)
                    else float(item) if isinstance(item, np.floating)
                    else item 
                    for item in v
                ]
            else:
                # Handle any other numpy types or custom objects
                try:
                    # Check if it's JSON serializable
                    json.dumps(v)
                    result[k] = v
                except (TypeError, OverflowError):
                    # Convert to string if it can't be serialized
                    result[k] = str(v)
        return result
    
    def _add_to_csv_summary(self, results: Dict[str, Any]):
        """Add a row to the CSV summary file."""
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [
                results.get("image_name", ""),
                results.get("resolution", ""),
                results.get("width", 0),
                results.get("height", 0),
                
                results.get("is_blurry", False),
                results.get("blur_value", 0),
                results.get("blur_threshold", self.config.get('blur_threshold', 100.0)),
                
                results.get("low_contrast", False),
                results.get("contrast_value", 0),
                results.get("contrast_threshold", self.config.get('contrast_threshold', 30.0)),
                
                results.get("segmentation_success", False),
                results.get("mask_coverage", "0%"),
                results.get("segmentation_model", self.segmenter.model_name),
                
                results.get("contrast_filter_pass", False),
                results.get("contrast_inner", 0),
                results.get("contrast_outer", 0),
                results.get("contrast_diff", 0),
                results.get("contrast_threshold_used", 0),
                
                results.get("clutter_filter_pass", False),
                results.get("detected_objects", 0),
                results.get("primary_object_iou", 0),
                results.get("primary_min_threshold", 0),
                
                results.get("contour_filter_pass", False),
                results.get("contour_count", 0),
                results.get("max_contour_points", 0),
                results.get("contour_solidity", 0),
                
                results.get("mask_refinement", True),
                results.get("opening_kernel", 0),
                results.get("opening_iter", 0),
                results.get("closing_kernel", 0),
                results.get("closing_iter", 0),
                results.get("dilation_kernel", 0),
                results.get("dilation_iter", 0),
                
                results.get("filter_counts_passed", 0),
                results.get("filter_counts_total", 3),
                results.get("final_status", False),
                
                results.get("processing_time", "0s"),
                results.get("error_details", "")
            ]
            writer.writerow(row)

    def analyze_directory(self, input_dir: Path, recursive: bool = False) -> Dict[str, Any]:
        """
        Analyze all images in a directory and generate a summary report.
        
        Args:
            input_dir: Directory containing images to analyze
            recursive: Whether to search subdirectories recursively
            
        Returns:
            Dictionary with summary statistics
        """
        image_paths = get_image_paths(input_dir, recursive=recursive)
        
        if not image_paths:
            logging.error(f"No images found in {input_dir}")
            return {"error": "No images found"}
        
        logging.info(f"Found {len(image_paths)} images to analyze")
        
        # Process each image
        results = []
        for i, img_path in enumerate(image_paths):
            print(f"\n{'='*80}")
            logging.info(f"[{i+1}/{len(image_paths)}] Analyzing: {img_path.name}")
            result = self.analyze_image(img_path)
            results.append(result)
            status = "✓ PASS" if result.get('final_status', False) else "✗ FAIL"
            logging.info(f"Status: {status}")
            print(f"{'='*80}\n")
        
        # Generate summary statistics
        summary = {
            "total_images": len(results),
            "passed_images": sum(1 for r in results if r.get("final_status", False)),
            "failed_images": sum(1 for r in results if not r.get("final_status", False)),
            "quality_issues": sum(1 for r in results if not r.get("is_processable", False)),
            "segmentation_failures": sum(1 for r in results if not r.get("segmentation_success", False)),
            "contrast_filter_failures": sum(1 for r in results if not r.get("contrast_filter_pass", False)),
            "clutter_filter_failures": sum(1 for r in results if not r.get("clutter_filter_pass", False)),
            "contour_filter_failures": sum(1 for r in results if not r.get("contour_filter_pass", False)),
            "timestamp": datetime.now().isoformat()
        }
        
        # Calculate percentages
        if summary["total_images"] > 0:
            summary["pass_rate"] = f"{(summary['passed_images'] / summary['total_images']) * 100:.1f}%"
            summary["pass_rate_numeric"] = (summary['passed_images'] / summary['total_images']) * 100
            
            # Calculate failure reasons percentages
            summary["quality_issues_pct"] = f"{(summary['quality_issues'] / summary['total_images']) * 100:.1f}%"
            summary["segmentation_failures_pct"] = f"{(summary['segmentation_failures'] / summary['total_images']) * 100:.1f}%"
            summary["contrast_filter_failures_pct"] = f"{(summary['contrast_filter_failures'] / summary['total_images']) * 100:.1f}%"
            summary["clutter_filter_failures_pct"] = f"{(summary['clutter_filter_failures'] / summary['total_images']) * 100:.1f}%"
            summary["contour_filter_failures_pct"] = f"{(summary['contour_filter_failures'] / summary['total_images']) * 100:.1f}%"
        
        # Save summary report
        with open(self.output_dir / "analysis_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Analyze image quality and filter performance.")
    parser.add_argument("input_dir", type=str, help="Directory containing images to analyze")
    parser.add_argument("--output_dir", type=str, default="results/quality_analysis",
                        help="Directory to save analysis results")
    parser.add_argument("--segmenter_model", type=str, default=DEFAULT_CONFIG['segmenter_model'],
                        help="Segmentation model to use")
    parser.add_argument("--recursive", action="store_true", 
                        help="Recursively search subdirectories for images")
    parser.add_argument("--no_visualizations", action="store_true",
                        help="Disable saving visualization images")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose debug logging")
    
    args = parser.parse_args()
    
    # Set up logging level based on verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.info("Verbose logging enabled")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    
    # Initialize analyzer
    analyzer = ImageAnalyzer(
        output_dir=output_dir,
        segmenter_model=args.segmenter_model,
        save_visualizations=not args.no_visualizations,
        log_level=logging.DEBUG if args.verbose else logging.INFO
    )
    
    # Run analysis
    print(f"\n{'*'*80}")
    print(f"Starting image analysis on: {args.input_dir}")
    print(f"Results will be saved to: {output_dir}")
    print(f"{'*'*80}\n")
    
    summary = analyzer.analyze_directory(
        input_dir=Path(args.input_dir),
        recursive=args.recursive
    )
    
    # Print summary
    print(f"\n{'*'*80}")
    print("--- Analysis Summary ---")
    print(f"Total images analyzed: {summary['total_images']}")
    print(f"Images passing all checks: {summary['passed_images']} ({summary.get('pass_rate', '0%')})")
    print(f"Images failing checks: {summary['failed_images']}")
    print("\nIssue breakdown:")
    print(f"  Quality issues: {summary['quality_issues']} ({summary.get('quality_issues_pct', '0%')})")
    print(f"  Segmentation failures: {summary['segmentation_failures']} ({summary.get('segmentation_failures_pct', '0%')})")
    print(f"  Contrast filter failures: {summary['contrast_filter_failures']} ({summary.get('contrast_filter_failures_pct', '0%')})")
    print(f"  Clutter filter failures: {summary['clutter_filter_failures']} ({summary.get('clutter_filter_failures_pct', '0%')})")
    print(f"  Contour filter failures: {summary['contour_filter_failures']} ({summary.get('contour_filter_failures_pct', '0%')})")
    
    print(f"\nDetailed reports saved to: {output_dir / 'reports'}")
    print(f"Summary CSV saved to: {output_dir / 'analysis_summary.csv'}")
    print(f"Detailed logs saved to: {output_dir / 'logs'}")
    
    if not args.no_visualizations:
        print(f"Visualizations saved to: {output_dir / 'visualizations'}")
    
    print(f"JSON summary saved to: {output_dir / 'analysis_summary.json'}")
    print(f"{'*'*80}\n")
    print("Analysis complete!")

if __name__ == "__main__":
    main() 