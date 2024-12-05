import csv
import os
from datetime import datetime

import cv2
from ultralytics import solutions

RESIZE_FACTOR = 0.5

# Add CSV setup
csv_file = open("detections.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Frame", "Timestamp", "Class", "Track ID", "X", "Y", "W", "H"])

# Add after CSV setup and before video capture
saved_track_ids = set()  # Keep track of which cars we've already saved

# Add near imports section
if not os.path.exists("detected_cars"):
    os.makedirs("detected_cars")

cap = cv2.VideoCapture("video/road.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (
    int(cap.get(x))
    for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)
)

region_points = [
    (1298, 796),
    (1444, 1428),
    (2374, 1436),
    (1554, 794),
    (1456, 752),
    (1284, 752),
]

video_writer = cv2.VideoWriter(
    "object_counting_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
)

trackzone = solutions.TrackZone(
    show=False,  # Changed from True to False
    region=region_points,
    model="yolo11n.pt",
)

cv2.namedWindow("TrackZone", cv2.WINDOW_NORMAL)
new_w = int(w * RESIZE_FACTOR)
new_h = int(h * RESIZE_FACTOR)
cv2.resizeWindow("TrackZone", new_w, new_h)

frame_count = 0
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print(
            "Video frame is empty or video processing has been successfully completed."
        )
        break

    # Process frame
    im0 = trackzone.trackzone(im0)

    # Log detections from trackzone instance
    if hasattr(trackzone, "boxes") and len(trackzone.boxes) > 0:
        print(f"Frame {frame_count}")
        print(f"Number of detections: {len(trackzone.boxes)}")

        for box, track_id, cls in zip(
            trackzone.boxes, trackzone.track_ids, trackzone.clss
        ):
            x1, y1, x2, y2 = box  # Boxes are already in x1,y1,x2,y2 format

            # Only save if we haven't seen this track_id before
            if track_id not in saved_track_ids:
                # Extract car ROI
                car_roi = im0[int(y1) : int(y2), int(x1) : int(x2)]

                # Skip if ROI is empty
                if car_roi.size == 0:
                    continue

                # Create filename with just track_id (since we only save once)
                filename = f"detected_cars/car_id{track_id}.jpg"

                try:
                    # Save the car image
                    cv2.imwrite(filename, car_roi)
                    # Mark this track_id as saved
                    saved_track_ids.add(track_id)
                    print(f"Saved first detection of car ID {track_id}")
                except Exception as e:
                    print(f"Error saving car image: {e}")

            print(
                f"Detection: Class={cls}, Track ID={track_id}, Box=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})"
            )

            csv_writer.writerow(
                [
                    frame_count,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    int(cls),
                    track_id,  # Using track_id instead of confidence
                    float(x1),
                    float(y1),
                    float(x2 - x1),  # width
                    float(y2 - y1),  # height
                ]
            )
            csv_file.flush()

    cv2.imshow("TrackZone", im0)
    video_writer.write(im0)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
video_writer.release()
csv_file.close()
cv2.destroyAllWindows()
