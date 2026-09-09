import cv2

from ml.tracking.tracker import ObjectTracker


def main():
    input_path = "data/demo/test.mp4"
    output_path = "data/demo/tracking_result.mp4"

    tracker = ObjectTracker()

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print(f"ERROR: Could not open video: {input_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height),
    )

    frame_number = 0

    print("\n===== IBVAP VIDEO TRACKING =====")

    while True:
        success, frame = cap.read()

        if not success:
            break

        frame_number += 1

        tracks = tracker.track(frame)

        for track in tracks:
            x1, y1, x2, y2 = track.bbox

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            label = (
                f"{track.class_name} "
                f"ID:{track.track_id} "
                f"{track.confidence:.2f}"
            )

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )

            print(
                f"Frame={frame_number:<5} "
                f"{track.class_name:<12} "
                f"ID={track.track_id:<4} "
                f"confidence={track.confidence:.2f}"
            )

        writer.write(frame)

    cap.release()
    writer.release()

    print("\n================================")
    print(f"Processed frames: {frame_number}")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()