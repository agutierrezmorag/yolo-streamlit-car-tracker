import cv2
from ultralytics.solutions.trackzone import TrackZone


class ROISelector:
    def __init__(self, max_width=1280, max_height=720):
        self.points = []
        self.drawing = False
        self.scale = 1.0
        self.max_width = max_width
        self.max_height = max_height

    def resize_frame(self, frame):
        height, width = frame.shape[:2]

        # Calculate scaling factor to fit within max dimensions
        scale_w = self.max_width / width
        scale_h = self.max_height / height
        self.scale = min(scale_w, scale_h, 1.0)

        if self.scale < 1.0:
            new_width = int(width * self.scale)
            new_height = int(height * self.scale)
            return cv2.resize(frame, (new_width, new_height))
        return frame

    def click_event(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Scale coordinates back to original size
            original_x = int(x / self.scale)
            original_y = int(y / self.scale)
            self.points.append((original_x, original_y))

    def select_roi(self, video_path):
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if not ret:
            raise Exception("Could not read video")

        # Resize frame for display
        display_frame = self.resize_frame(frame)

        win_name = "Select ROI - Click points, press 'q' when done"
        cv2.namedWindow(win_name)
        cv2.setMouseCallback(win_name, self.click_event)

        while True:
            display_copy = display_frame.copy()

            # Draw points and lines (scaled for display)
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


# Usage remains the same
if __name__ == "__main__":
    roi_selector = ROISelector(max_width=1280, max_height=720)  # Set max dimensions
    video_path = "video/road.mp4"
    polygon_points = roi_selector.select_roi(video_path)
    tracker = TrackZone(region=polygon_points)

    # Rest of the code remains unchanged...

    # Process video with selected ROI
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = tracker.trackzone(frame)
        cv2.imshow("Tracking", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
