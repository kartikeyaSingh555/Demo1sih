import cv2

from ml.detection.detector import ObjectDetector


def main():
    detector = ObjectDetector()

    image_path = "data/demo/test.jpg"
    output_path = "data/demo/detection_result.jpg"

    frame = cv2.imread(image_path)

    if frame is None:
        print(f"ERROR: Could not load image: {image_path}")
        return

    detections = detector.detect(frame)

    print("\n===== IBVAP DETECTION RESULT =====")

    for detection in detections:
        print(
            f"{detection.class_name:<12} "
            f"confidence={detection.confidence:.2f} "
            f"bbox={detection.bbox}"
        )

        x1, y1, x2, y2 = detection.bbox

        # Draw bounding box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        # Label
        label = (
            f"{detection.class_name} "
            f"{detection.confidence:.2f}"
        )

        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    print("==================================")

    cv2.imwrite(output_path, frame)

    print(f"\nDetection image saved to:")
    print(output_path)


if __name__ == "__main__":
    main()