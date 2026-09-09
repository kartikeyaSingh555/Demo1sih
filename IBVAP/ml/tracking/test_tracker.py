import cv2

from ml.tracking.tracker import ObjectTracker


def main():
    tracker = ObjectTracker()

    image_path = "data/demo/test.jpg"
    output_path = "data/demo/tracking_result.jpg"

    frame = cv2.imread(image_path)

    if frame is None:
        print(f"ERROR: Could not load image: {image_path}")
        return

    tracks = tracker.track(frame)

    print("\n===== IBVAP TRACKING RESULT =====")

    for track in tracks:
        print(
            f"{track.class_name:<12} "
            f"ID={track.track_id:<4} "
            f"confidence={track.confidence:.2f} "
            f"bbox={track.bbox}"
        )

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

    print("=================================")

    cv2.imwrite(output_path, frame)

    print(f"\nTracking image saved to:")
    print(output_path)


if __name__ == "__main__":
    main()