import cv2

from ml.drone.detector import DroneDetector


IMAGE_PATH = "data/demo/drone_test.jpg"
OUTPUT_PATH = "data/demo/drone_detection_result.jpg"


def main():

    print("\n======================================")
    print("        IBVAP DRONE DETECTION TEST")
    print("======================================")

    print("\n[1/3] Loading drone detector...")

    detector = DroneDetector()

    print("\n[2/3] Loading test image...")

    frame = cv2.imread(IMAGE_PATH)

    if frame is None:
        print(
            f"ERROR: Could not load image: "
            f"{IMAGE_PATH}"
        )
        return

    print(
        f"Image loaded: {IMAGE_PATH}"
    )

    print("\n[3/3] Detecting drones...")

    detections = detector.detect(frame)

    print(
        "\n========== DRONE RESULTS =========="
    )

    print(
        f"Drones detected: {len(detections)}"
    )

    for index, detection in enumerate(
        detections,
        start=1,
    ):

        print(
            f"\nDrone {index}"
        )

        print(
            f"Class       : "
            f"{detection.class_name}"
        )

        print(
            f"Confidence  : "
            f"{detection.confidence:.4f}"
        )

        print(
            f"BBox        : "
            f"{detection.bbox}"
        )

        x1, y1, x2, y2 = detection.bbox

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            3,
        )

        label = (
            f"{detection.class_name} "
            f"{detection.confidence:.2f}"
        )

        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
        )

    print(
        "\n===================================="
    )

    cv2.imwrite(
        OUTPUT_PATH,
        frame,
    )

    print(
        "\nResult image saved to:"
    )

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()