import csv
import os
from datetime import datetime

import cv2
from ultralytics import solutions

from constants import REGION_POINTS, RESIZE_FACTOR, YOLO_MODELS


def create_directories(model_name):
    """Create output directories for a specific model"""
    base_dir = f"output_{model_name.replace('.pt','')}"
    dirs = {
        "base": base_dir,
        "cars": os.path.join(base_dir, "detected_cars"),
        "csv": os.path.join(base_dir, "data"),
    }
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    return dirs


def process_video(model_name):
    """Process video with specified YOLO model"""
    print(f"\nProcessing with model: {model_name}")

    # Create directories
    dirs = create_directories(model_name)

    # Setup CSV
    csv_path = os.path.join(dirs["csv"], "detections.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Frame", "Timestamp", "Class", "Track ID", "X", "Y", "W", "H"])

    # Track saved IDs
    saved_track_ids = set()

    # Video setup
    cap = cv2.VideoCapture("video/road.mp4")
    assert cap.isOpened(), "Error reading video file"
    w, h, fps = (
        int(cap.get(x))
        for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)
    )

    # Video writer
    video_path = os.path.join(
        dirs["base"], f"output_{model_name.replace('.pt','')}.avi"
    )
    video_writer = cv2.VideoWriter(
        video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )

    # Initialize tracker
    trackzone = solutions.TrackZone(
        show=False,
        region=REGION_POINTS,
        model=model_name,
    )

    # Window setup
    window_name = f"TrackZone_{model_name.replace('.pt','')}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    new_w = int(w * RESIZE_FACTOR)
    new_h = int(h * RESIZE_FACTOR)
    cv2.resizeWindow(window_name, new_w, new_h)

    frame_count = 0
    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print(f"Completed processing with {model_name}")
            break

        # Process frame
        im0 = trackzone.trackzone(im0)

        # Handle detections
        if hasattr(trackzone, "boxes") and len(trackzone.boxes) > 0:
            for box, track_id, cls in zip(
                trackzone.boxes, trackzone.track_ids, trackzone.clss
            ):
                x1, y1, x2, y2 = box

                # Save first detection of each car
                if track_id not in saved_track_ids:
                    car_roi = im0[int(y1) : int(y2), int(x1) : int(x2)]
                    if car_roi.size > 0:
                        filename = os.path.join(dirs["cars"], f"car_id{track_id}.jpg")
                        try:
                            cv2.imwrite(filename, car_roi)
                            saved_track_ids.add(track_id)
                        except Exception as e:
                            print(f"Error saving car image: {e}")

                # Write to CSV
                csv_writer.writerow(
                    [
                        frame_count,
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        int(cls),
                        track_id,
                        float(x1),
                        float(y1),
                        float(x2 - x1),
                        float(y2 - y1),
                    ]
                )
                csv_file.flush()

        cv2.imshow(window_name, im0)
        video_writer.write(im0)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup
    cap.release()
    video_writer.release()
    csv_file.close()
    cv2.destroyAllWindows()


def main():
    """Run processing for all YOLO models"""
    for model in YOLO_MODELS:
        process_video(model)


if __name__ == "__main__":
    main()
