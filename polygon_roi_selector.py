import cv2

from constants import RESIZE_FACTOR


class ROISelector:
    def __init__(self):
        self.points = []
        self.drawing = False
        self.scale = RESIZE_FACTOR

    def resize_frame(self, frame):
        """Resize frame using constant resize factor"""
        height, width = frame.shape[:2]
        new_width = int(width * self.scale)
        new_height = int(height * self.scale)
        return cv2.resize(frame, (new_width, new_height))

    def click_event(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Scale coordinates back to original size
            original_x = int(x / self.scale)
            original_y = int(y / self.scale)
            self.points.append((original_x, original_y))

    def save_to_constants(self):
        """Save ROI points to constants.py"""
        points_str = "[\n    " + ",\n    ".join([str(p) for p in self.points]) + "\n]"

        # Read existing content to preserve RESIZE_FACTOR
        with open("constants.py", "r") as f:
            lines = f.readlines()
            resize_factor_line = next(
                (line for line in lines if "RESIZE_FACTOR" in line), None
            )

        constants_content = f"""# Region points for vehicle detection
# Format: [(x1,y1), (x2,y2), ...]
REGION_POINTS = {points_str}

{resize_factor_line or f'RESIZE_FACTOR = {RESIZE_FACTOR}'}"""

        with open("constants.py", "w") as f:
            f.write(constants_content)

    def select_roi(self, video_path):
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if not ret:
            raise Exception("Could not read video")

        # Resize frame for display
        display_frame = self.resize_frame(frame)

        win_name = "Select ROI - Click points, press 'q' when done"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win_name, self.click_event)

        while True:
            display_copy = display_frame.copy()

            # Draw points and lines
            for i, point in enumerate(self.points):
                # Scale points for display
                display_point = (int(point[0] * self.scale), int(point[1] * self.scale))
                cv2.circle(display_copy, display_point, 3, (0, 255, 0), -1)

                if i > 0:
                    prev_point = (
                        int(self.points[i - 1][0] * self.scale),
                        int(self.points[i - 1][1] * self.scale),
                    )
                    cv2.line(display_copy, prev_point, display_point, (0, 255, 0), 2)

            if len(self.points) > 2:
                first_point = (
                    int(self.points[0][0] * self.scale),
                    int(self.points[0][1] * self.scale),
                )
                last_point = (
                    int(self.points[-1][0] * self.scale),
                    int(self.points[-1][1] * self.scale),
                )
                cv2.line(display_copy, last_point, first_point, (0, 255, 0), 2)

            cv2.imshow(win_name, display_copy)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()
        cap.release()
        return self.points


if __name__ == "__main__":
    roi_selector = ROISelector()
    video_path = "video/road.mp4"

    # Select and save ROI
    polygon_points = roi_selector.select_roi(video_path)
    roi_selector.save_to_constants()
    print(f"ROI saved with {len(polygon_points)} points!")
