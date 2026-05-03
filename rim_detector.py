import cv2
import numpy as np
import os


class RimDetector:
    def __init__(self, debug_mode=False):
        # HSV ranges for orange/red rim (assuming BGR input)
        self.lower_mask = np.array([0, 40, 40])
        self.upper_mask = np.array([20, 255, 255])
        self.lower_orange = np.array([160, 40, 40])
        self.upper_orange = np.array([180, 255, 255])

        self.initialization_frames = 20
        self.frame_count = 0
        self.total_center_x = 0.0
        self.total_center_y = 0.0
        self.total_max_radius = 0.0
        self.total_min_radius = 0.0
        self.total_avg_radius = 0.0
        self.total_area = 0.0
        self.total_angle = 0.0

        # Final fixed position (None until initialization complete)
        self.fixed_ellipse = None

        self.kernel = np.ones((3, 3), np.uint8)
        self.edge_kernel = np.ones((2, 2), np.uint8)

        # CLAHE instances — created once, reused every frame
        self.clahe_gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.clahe_value = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))

        # Circularity constraints
        self.max_aspect_ratio = 2.5
        self.min_aspect_ratio = 0.4
        self.min_radius = 10
        self.max_radius = 400

        # Debug mode
        self.debug_mode = debug_mode
        self.debug_dir = "debug_frames"
        if self.debug_mode:
            os.makedirs(self.debug_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def is_valid_rim(self, ellipse, contour=None, debug=False):
        """Check whether the detected ellipse meets rim geometry criteria."""
        _, (major_axis, minor_axis), _ = ellipse

        if major_axis == 0 or minor_axis == 0:
            return False

        aspect_ratio = (
            minor_axis / major_axis
            if major_axis > minor_axis
            else major_axis / minor_axis
        )
        avg_radius = (major_axis + minor_axis) / 4.0

        if debug:
            print(
                f"    Aspect ratio: {aspect_ratio:.3f} "
                f"(must be {self.min_aspect_ratio}-{self.max_aspect_ratio})"
            )
            print(
                f"    Avg radius: {avg_radius:.1f} "
                f"(must be {self.min_radius}-{self.max_radius})"
            )

        if not (
            self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio
            and self.min_radius <= avg_radius <= self.max_radius
        ):
            if debug:
                if not (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
                    print("    ✗ FAILED: Aspect ratio out of range")
                if not (self.min_radius <= avg_radius <= self.max_radius):
                    print("    ✗ FAILED: Radius out of range")
            return False

        if contour is not None:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)

            if perimeter > 0:
                circularity = 4.0 * np.pi * area / (perimeter * perimeter)
                if debug:
                    print(f"    Circularity: {circularity:.3f} (must be >= 0.2)")
                if circularity < 0.2:
                    if debug:
                        print("    ✗ FAILED: Circularity too low")
                    return False

            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = area / hull_area
                if debug:
                    print(f"    Solidity: {solidity:.3f} (must be >= 0.4)")
                if solidity < 0.4:
                    if debug:
                        print("    ✗ FAILED: Solidity too low")
                    return False

        if debug:
            print("    ✓ PASSED all checks")
        return True

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------

    def _save_debug(self, image, name, frame_num):
        if self.debug_mode and frame_num <= 5:
            path = os.path.join(self.debug_dir, f"frame_{frame_num:03d}_{name}.jpg")
            cv2.imwrite(path, image)

    # ------------------------------------------------------------------
    # Enhanced edge + colour pipeline
    # ------------------------------------------------------------------

    def _build_edge_mask(self, gray):
        """CLAHE-enhanced adaptive Canny edges."""
        enhanced = self.clahe_gray.apply(gray)

        # Adaptive Canny thresholds from median intensity
        med = float(np.median(enhanced))
        lo = int(max(0, 0.55 * med))
        hi = int(min(255, 1.4 * med))
        edges = cv2.Canny(enhanced, lo, hi)

        # Dilate to bridge small gaps in the rim contour
        edges = cv2.dilate(edges, self.edge_kernel, iterations=2)
        return edges

    def _build_colour_mask(self, bgr):
        """Multi-channel colour mask with CLAHE on the V channel."""
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # Enhance the Value channel so dim / unevenly-lit rims pop
        h, s, v = cv2.split(hsv)
        v = self.clahe_value.apply(v)
        hsv = cv2.merge([h, s, v])

        hsv = cv2.GaussianBlur(hsv, (3, 3), 0)

        mask_lo = cv2.inRange(hsv, self.lower_mask, self.upper_mask)
        mask_hi = cv2.inRange(hsv, self.lower_orange, self.upper_orange)
        colour = cv2.bitwise_or(mask_lo, mask_hi)

        # Clean up
        colour = cv2.dilate(colour, self.kernel, iterations=1)
        colour = cv2.morphologyEx(colour, cv2.MORPH_CLOSE, self.kernel)
        colour = cv2.morphologyEx(colour, cv2.MORPH_OPEN, self.kernel)
        return colour

    # ------------------------------------------------------------------
    # Main detection entry point
    # ------------------------------------------------------------------

    def detect_rim(self, frame):
        """
        Detect the basketball rim using combined edge + colour contour analysis.

        Args:
            frame: **BGR** image (numpy.ndarray).

        Returns:
            ((cx, cy), (major, minor), angle) or None.
        """
        current_frame = self.frame_count + 1

        # Once initialised, always return the locked position
        if self.fixed_ellipse is not None:
            return self.fixed_ellipse

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = self._build_edge_mask(gray)
        colour = self._build_colour_mask(frame)

        # Combine: keep only edges that overlap with colour
        combined = cv2.bitwise_and(colour, edges)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, self.kernel, iterations=2)

        self._save_debug(edges, "edges", current_frame)
        self._save_debug(colour, "colour", current_frame)
        self._save_debug(combined, "combined", current_frame)

        # Contour search
        contours, _ = cv2.findContours(
            combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        valid_ellipse = None
        largest_area = 0

        for contour in contours:
            if len(contour) < 5:
                continue
            try:
                ellipse = cv2.fitEllipse(contour)
            except cv2.error:
                continue
            area = cv2.contourArea(contour)
            if self.is_valid_rim(ellipse, contour) and area > largest_area:
                largest_area = area
                valid_ellipse = ellipse

        # Debug: first frame diagnostics
        if self.debug_mode and current_frame == 1:
            print(f"\n=== FRAME {current_frame} DEBUG ===")
            print(f"Contours: {len(contours)}")
            if valid_ellipse is not None:
                print(f"Best ellipse area: {largest_area:.0f}")

        if valid_ellipse is None:
            if current_frame == 1:
                print(f"!!! No valid rim in frame {current_frame} !!!")
            return None

        # Accumulate during initialisation window
        if self.frame_count < self.initialization_frames:
            cx, cy = valid_ellipse[0]
            major, minor = valid_ellipse[1]
            angle = valid_ellipse[2]

            self.total_center_x += cx
            self.total_center_y += cy
            self.total_max_radius += max(major, minor) / 2.0
            self.total_min_radius += min(major, minor) / 2.0
            self.total_avg_radius += (major + minor) / 4.0
            self.total_area += np.pi * (major / 2.0) * (minor / 2.0)
            self.total_angle += angle
            self.frame_count += 1

            if self.frame_count == self.initialization_frames:
                n = float(self.initialization_frames)
                avg_center = (self.total_center_x / n, self.total_center_y / n)
                avg_axes = (
                    2.0 * self.total_avg_radius / n,
                    2.0 * self.total_min_radius / n,
                )
                avg_angle = self.total_angle / n
                self.fixed_ellipse = (avg_center, avg_axes, avg_angle)
                print(
                    f"\n=== RIM LOCKED ===\n"
                    f"Center: ({avg_center[0]:.1f}, {avg_center[1]:.1f})\n"
                    f"Radii : ({avg_axes[0]/2:.1f}, {avg_axes[1]/2:.1f})\n"
                    f"==================\n"
                )
                return self.fixed_ellipse

            return valid_ellipse  # interim result during init

        return None

    # ------------------------------------------------------------------
    # Accessors / drawing
    # ------------------------------------------------------------------

    def get_rim_parameters(self):
        """Return fixed rim parameters dict, or None if not yet locked."""
        if self.fixed_ellipse is None:
            return None
        cx, cy = self.fixed_ellipse[0]
        major, minor = self.fixed_ellipse[1]
        return {
            "center_x": int(cx),
            "center_y": int(cy),
            "max_radius": int(max(major, minor) / 2),
            "min_radius": int(min(major, minor) / 2),
            "avg_radius": int((major + minor) / 4),
            "area": int(np.pi * (major / 2) * (minor / 2)),
            "initialization_complete": self.frame_count >= self.initialization_frames,
        }

    def draw_rim(self, frame, ellipse):
        """Draw the rim ellipse and centre dot on *frame* in-place."""
        if ellipse is not None:
            cv2.ellipse(frame, ellipse, (0, 255, 125), 3)
            center = (int(ellipse[0][0]), int(ellipse[0][1]))
            cv2.circle(frame, center, 3, (0, 0, 255), -1)
