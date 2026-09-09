import cv2

from ml.anpr.plate_detector import LicensePlateDetector


def main():
    image_path = "data/demo/test.jpg"
    output_path = "data/demo/plate_detection_result.jpg"

    frame = cv2.imread(image_path)

    if frame is None:
        print(f"ERROR: Could not load image: {image_path}")
        return

    detector = LicensePlateDetector()

    detections = detector.detect(frame)

    print("\n===== IBVAP LICENSE PLATE DETECTION =====")

    if not detections:
        print("No license plates detected.")

    for index, detection in enumerate(detections, start=1):
        print(
            f"Plate {index:<3} "
            f"confidence={detection.confidence:.2f} "
            f"bbox={detection.bbox}"
        )

        x1, y1, x2, y2 = detection.bbox

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        label = f"plate {detection.confidence:.2f}"

        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
        )

    print("==========================================")

    cv2.imwrite(output_path, frame)

    print(f"\nPlate detection image saved to:")
    print(output_path)


if __name__ == "__main__":
    main()