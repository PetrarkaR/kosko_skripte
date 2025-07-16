import cv2
import numpy as np

class RimDetector:
    def __init__(self):
        self.lower_mask = np.array([0, 60, 60])
        self.upper_mask = np.array([40, 255, 255])
        self.lower_orange = np.array([170, 50, 50])
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

        # Circularity constraints
        self.max_aspect_ratio = 1.6 
        self.min_aspect_ratio = 0.7  
        self.min_radius = 20  # Minimum radius in pixels
        self.max_radius = 600  # Maximum radius in pixels

    def is_valid_rim(self, ellipse):
        """
        Checks if the detected ellipse meets our circularity criteria.
        
        Args:
            ellipse: The detected ellipse parameters.
            
        Returns:
            bool: True if the ellipse meets our criteria, False otherwise.
        """
        _, (major_axis, minor_axis), _ = ellipse
        
        # Calculate aspect ratio
        aspect_ratio = minor_axis / major_axis if major_axis > minor_axis else major_axis / minor_axis
        
        # Check if the size is within acceptable range
        avg_radius = (major_axis + minor_axis) / 4
        
        return (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio and
                self.min_radius <= avg_radius <= self.max_radius)

    def detect_rim(self, frame):
        """
        Detects the basketball rim in a given frame. During the first 20 frames,
        it accumulates data for averaging. After that, it uses the fixed position.

        Args:
            frame (numpy.ndarray): BGR image frame.

        Returns:
            tuple: ((center_x, center_y), (major_axis, minor_axis), angle) if rim found, else None.
        """
        # If initialization is complete, return the fixed position
        if self.fixed_ellipse is not None:
            return self.fixed_ellipse

        # Process frame for rim detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv = cv2.GaussianBlur(hsv,(3,3),0)
        hsv = cv2.medianBlur(hsv,1)

        # Create a binary mask for orange
        mask1 = cv2.inRange(hsv, self.lower_mask, self.upper_mask)
        mask2 = cv2.inRange(hsv, self.lower_orange, self.upper_orange)
        mask = cv2.bitwise_or(mask1, mask2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Track the largest valid rim contour
        valid_ellipse = None
        largest_valid_area = 0

        for contour in contours:
            if len(contour) >= 5:
                try:
                    ellipse = cv2.fitEllipse(contour)
                    area = cv2.contourArea(contour)
                    
                    # Check if this ellipse meets our circularity criteria
                    if self.is_valid_rim(ellipse) and area > largest_valid_area:
                        largest_valid_area = area
                        valid_ellipse = ellipse
                except:
                    continue

        # If a valid ellipse is found, process it
        if valid_ellipse is not None:
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
                    return self.fixed_ellipse

                return valid_ellipse  # Return current detection during initialization

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