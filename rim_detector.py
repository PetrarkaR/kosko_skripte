import cv2
import numpy as np
import os

class RimDetector:
    def __init__(self, debug_mode=True):
        self.lower_mask = np.array([0, 40, 40])  # More relaxed saturation/value
        self.upper_mask = np.array([20, 255, 255])
        self.lower_orange = np.array([160, 40, 40])  # More relaxed
        self.upper_orange = np.array([180, 255, 255])

        self.initialization_frames = 20
        self.frame_count = 0
        self.total_center_x = 0
        self.total_center_y = 0
        self.total_max_radius = 0
        self.total_min_radius = 0
        self.total_avg_radius = 0
        self.total_area = 0
        self.total_angle = 0

        # Final fixed position (None until initialization complete)
        self.fixed_ellipse = None
        self.kernel = np.ones((3,3), np.uint16)
        self.edge_kernel = np.ones((2,2), np.uint8)

        # Relaxed circularity constraints
        self.max_aspect_ratio = 2.5  # More lenient (was 1.6)
        self.min_aspect_ratio = 0.4  # More lenient (was 0.7) - allows 0.658
        self.min_radius = 10  # Smaller minimum (was 20)
        self.max_radius = 400  # Larger maximum (was 200) - allows 222.6
        
        # Debug mode
        self.debug_mode = debug_mode
        self.debug_dir = "debug_frames"
        if self.debug_mode:
            os.makedirs(self.debug_dir, exist_ok=True)

    def is_valid_rim(self, ellipse, contour=None, debug=False):
        """
        Checks if the detected ellipse meets our circularity criteria.
        
        Args:
            ellipse: The detected ellipse parameters.
            contour: Optional contour for additional validation.
            debug: If True, print detailed validation info
            
        Returns:
            bool: True if the ellipse meets our criteria, False otherwise.
        """
        _, (major_axis, minor_axis), _ = ellipse
        
        # Calculate aspect ratio
        aspect_ratio = minor_axis / major_axis if major_axis > minor_axis else major_axis / minor_axis
        
        # Check if the size is within acceptable range
        avg_radius = (major_axis + minor_axis) / 4
        
        if debug:
            print(f"    Aspect ratio: {aspect_ratio:.3f} (must be {self.min_aspect_ratio}-{self.max_aspect_ratio})")
            print(f"    Avg radius: {avg_radius:.1f} (must be {self.min_radius}-{self.max_radius})")
        
        # Basic size and aspect ratio check
        if not (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio and
                self.min_radius <= avg_radius <= self.max_radius):
            if debug:
                if not (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
                    print(f"    ✗ FAILED: Aspect ratio out of range")
                if not (self.min_radius <= avg_radius <= self.max_radius):
                    print(f"    ✗ FAILED: Radius out of range")
            return False
        
        # Additional geometric validation if contour is provided (more relaxed)
        if contour is not None:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # Circularity test (closer to 1.0 = more circular)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if debug:
                    print(f"    Circularity: {circularity:.3f} (must be >= 0.2)")
                if circularity < 0.2:  # Very relaxed (was 0.5)
                    if debug:
                        print(f"    ✗ FAILED: Circularity too low")
                    return False
            
            # Solidity test (ratio of contour area to convex hull area)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = area / hull_area
                if debug:
                    print(f"    Solidity: {solidity:.3f} (must be >= 0.4)")
                if solidity < 0.4:  # Very relaxed (was 0.7)
                    if debug:
                        print(f"    ✗ FAILED: Solidity too low")
                    return False
        
        if debug:
            print(f"    ✓ PASSED all checks")
        return True

    def save_debug_image(self, image, name, frame_num):
        """Save debug image with frame number"""
        if self.debug_mode and frame_num <= 5:  # Save first 5 frames
            filename = os.path.join(self.debug_dir, f"frame_{frame_num:03d}_{name}.jpg")
            cv2.imwrite(filename, image)

    def detect_rim(self, frame):
        """
        Detects the basketball rim using combined edge+color mask with contour analysis.

        Args:
            frame (numpy.ndarray): BGR image frame.

        Returns:
            tuple: ((center_x, center_y), (major_axis, minor_axis), angle) if rim found, else None.
        """
        current_frame = self.frame_count + 1
        
        # If initialization is complete, return the fixed position
        if self.fixed_ellipse is not None:
            return self.fixed_ellipse

        # Save original frame
        #self.save_debug_image(frame, "01_original", current_frame)

        # === STEP 1: Edge Detection ===
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #self.save_debug_image(gray, "02_gray", current_frame)
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        #self.save_debug_image(gray, "03_clahe", current_frame)
        
        # Adaptive Canny edge detection
        v = np.median(gray)
        lower = int(max(0, 0.66 * v))
        upper = int(min(255, 1.33 * v))
        edges = cv2.Canny(gray, lower, upper)
        #self.save_debug_image(edges, "04_edges", current_frame)
        
        # Dilate edges to make them more connected
        edges_dilated = cv2.dilate(edges, self.edge_kernel, iterations=2)
        #self.save_debug_image(edges_dilated, "05_edges_dilated", current_frame)
        
        # === STEP 2: Color Mask ===
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv = cv2.GaussianBlur(hsv, (3,3), 0)
        hsv = cv2.medianBlur(hsv, 1)

        # Create binary mask for orange
        mask1 = cv2.inRange(hsv, self.lower_mask, self.upper_mask)
        mask2 = cv2.inRange(hsv, self.lower_orange, self.upper_orange)
        #self.save_debug_image(mask1, "06_mask1_yellow", current_frame)
        #self.save_debug_image(mask2, "07_mask2_orange", current_frame)
        
        color_mask = cv2.bitwise_or(mask1, mask2)
        #self.save_debug_image(color_mask, "08_color_mask_combined", current_frame)
        
        color_mask = cv2.dilate(color_mask, self.kernel)
        #self.save_debug_image(color_mask, "09_color_mask_dilated", current_frame)
        
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, self.kernel)
        #self.save_debug_image(color_mask, "10_color_mask_closed", current_frame)
        
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, self.kernel)
        #self.save_debug_image(color_mask, "11_color_mask_opened", current_frame)
        
        # === STEP 3: Combine Edges + Color ===
        combined_mask = cv2.bitwise_and(color_mask, edges_dilated)
        #self.save_debug_image(combined_mask, "12_combined_mask", current_frame)
        
        # Apply additional morphological operations to clean up the combined mask
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        #self.save_debug_image(combined_mask, "13_combined_final", current_frame)
        
        # === STEP 4: Find Contours in Combined Mask ===
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw all contours for debugging
        if self.debug_mode and current_frame <= 5:
            debug_contours = frame.copy()
            cv2.drawContours(debug_contours, contours, -1, (0, 255, 0), 2)
            self.save_debug_image(debug_contours, "14_all_contours", current_frame)

        # Track the largest valid rim contour
        valid_ellipse = None
        largest_valid_area = 0
        all_ellipses = []

        for idx, contour in enumerate(contours):
            if len(contour) >= 5:
                try:
                    ellipse = cv2.fitEllipse(contour)
                    area = cv2.contourArea(contour)
                    
                    # Store all ellipses for debugging
                    all_ellipses.append((ellipse, area, self.is_valid_rim(ellipse, contour)))
                    
                    # Check if this ellipse meets our circularity criteria
                    if self.is_valid_rim(ellipse, contour) and area > largest_valid_area:
                        largest_valid_area = area
                        valid_ellipse = ellipse
                except:
                    continue

        # Debug: draw all detected ellipses
        if self.debug_mode and current_frame <= 5:
            debug_ellipses = frame.copy()
            for ellipse, area, is_valid in all_ellipses:
                color = (0, 255, 0) if is_valid else (0, 0, 255)  # Green if valid, red if not
                cv2.ellipse(debug_ellipses, ellipse, color, 2)
                center = tuple(map(int, ellipse[0]))
                cv2.putText(debug_ellipses, f"A:{int(area)}", center, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            self.save_debug_image(debug_ellipses, "15_all_ellipses", current_frame)
            
            # Print debug info
            if current_frame == 1:
                print(f"\n=== FRAME {current_frame} DEBUG INFO ===")
                print(f"Total contours found: {len(contours)}")
                print(f"Total ellipses fitted: {len(all_ellipses)}")
                print(f"Valid ellipses: {sum(1 for _, _, is_valid in all_ellipses if is_valid)}")
                if all_ellipses:
                    print("\nEllipse validation details:")
                    for i, (ellipse, area, is_valid) in enumerate(all_ellipses):
                        center, axes, angle = ellipse
                        print(f"\n  Ellipse {i+1}:")
                        print(f"    Area: {area:.1f}, Center: ({center[0]:.1f}, {center[1]:.1f})")
                        print(f"    Axes: ({axes[0]:.1f}, {axes[1]:.1f}), Angle: {angle:.1f}°")
                        # Re-validate with debug output
                        contour = contours[i] if i < len(contours) and len(contours[i]) >= 5 else None
                        self.is_valid_rim(ellipse, contour, debug=True)

        # If a valid ellipse is found, process it
        if valid_ellipse is not None:
            # Draw the selected ellipse
            if self.debug_mode and current_frame <= 5:
                debug_selected = frame.copy()
                cv2.ellipse(debug_selected, valid_ellipse, (0, 255, 125), 3)
                center = tuple(map(int, valid_ellipse[0]))
                cv2.circle(debug_selected, center, 5, (0, 0, 255), -1)
                self.save_debug_image(debug_selected, "16_selected_ellipse", current_frame)
            
            # During initialization period, accumulate measurements
            if self.frame_count < self.initialization_frames:
                center_x, center_y = valid_ellipse[0]
                major_axis, minor_axis = valid_ellipse[1]
                angle = valid_ellipse[2]

                self.total_center_x += center_x
                self.total_center_y += center_y
                self.total_max_radius += max(major_axis, minor_axis) / 2
                self.total_min_radius += min(major_axis, minor_axis) / 2
                self.total_avg_radius += (major_axis + minor_axis) / 4
                self.total_area += np.pi * (major_axis / 2) * (minor_axis / 2)
                self.total_angle += angle
                self.frame_count += 1

                # If this is the last initialization frame, compute the fixed position
                if self.frame_count == self.initialization_frames:
                    avg_center = (
                        self.total_center_x / self.initialization_frames,
                        self.total_center_y / self.initialization_frames
                    )
                    avg_axes = (
                        2 * self.total_avg_radius / self.initialization_frames,
                        2 * self.total_min_radius / self.initialization_frames
                    )
                    avg_angle = self.total_angle / self.initialization_frames

                    self.fixed_ellipse = (avg_center, avg_axes, avg_angle)
                    print(f"\n=== RIM LOCKED ===")
                    print(f"Center: ({avg_center[0]:.1f}, {avg_center[1]:.1f})")
                    print(f"Radii: ({avg_axes[0]/2:.1f}, {avg_axes[1]/2:.1f})")
                    print(f"==================\n")
                    return self.fixed_ellipse

                return valid_ellipse  # Return current detection during initialization
        else:
            if current_frame == 1:
                print(f"\n!!! NO VALID RIM DETECTED IN FRAME {current_frame} !!!")
                print("Check debug_frames folder for diagnostic images.\n")

        return None  # No valid rim detected

    def get_rim_parameters(self):
        """
        Returns the fixed rim parameters after initialization is complete.

        Returns:
            dict: Dictionary containing rim center, radii, and area, or None if initialization isn't complete.
        """
        if self.fixed_ellipse is None:
            return None

        center_x, center_y = self.fixed_ellipse[0]
        major_axis, minor_axis = self.fixed_ellipse[1]

        rim_data = {
            "center_x": int(center_x),
            "center_y": int(center_y),
            "max_radius": int(max(major_axis, minor_axis) / 2),
            "min_radius": int(min(major_axis, minor_axis) / 2),
            "avg_radius": int((major_axis + minor_axis) / 4),
            "area": int(np.pi * (major_axis / 2) * (minor_axis / 2)),
            "initialization_complete": self.frame_count >= self.initialization_frames
        }
        return rim_data

    def draw_rim(self, frame, ellipse):
        """
        Draws the detected rim ellipse on the given frame.

        Args:
            frame (numpy.ndarray): The image frame.
            ellipse (tuple): The fitted ellipse parameters ((center_x, center_y), (major_axis, minor_axis), angle).
        """
        if ellipse is not None:
            cv2.ellipse(frame, ellipse, (0, 255, 125), 3)
            center = tuple(map(int, ellipse[0]))
            cv2.circle(frame, center, 3, (0, 0, 255), -1)  # Draw center point
