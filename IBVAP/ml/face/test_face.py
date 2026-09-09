import cv2

from ml.face.face_detector import FaceDetector


def main():

    image_path = "data/demo/face_test.jpg"
    output_path = "data/demo/face_detection_result.jpg"

    print("\n======================================")
    print("       IBVAP FACE DETECTION TEST")
    print("======================================")

    print(
        f"\nInput image: {image_path}"
    )

    frame = cv2.imread(
        image_path
    )

    if frame is None:

        print(
            "\nERROR: Could not load image."
        )

        print(
            "Please place a clear face image at:"
        )

        print(
            image_path
        )

        return

    print(
        f"Image size: "
        f"{frame.shape[1]}x"
        f"{frame.shape[0]}"
    )

    # -------------------------------------------------
    # Load InsightFace
    # -------------------------------------------------

    print(
        "\nLoading InsightFace..."
    )

    detector = FaceDetector()

    # -------------------------------------------------
    # Detect faces
    # -------------------------------------------------

    print(
        "\nDetecting faces..."
    )

    detections = detector.detect(
        frame
    )

    print(
        f"Faces detected: "
        f"{len(detections)}"
    )

    # -------------------------------------------------
    # Results
    # -------------------------------------------------

    print(
        "\n========== FACE RESULTS =========="
    )

    if not detections:

        print(
            "No faces detected."
        )

    for index, face in enumerate(
        detections,
        start=1,
    ):

        print(
            f"\nFace {index}"
        )

        print(
            f"Bounding box : "
            f"{face.bbox}"
        )

        print(
            f"Detection confidence : "
            f"{face.confidence:.4f}"
        )

        print(
            f"Embedding dimensions : "
            f"{face.embedding.shape}"
        )

        print(
            f"Embedding norm : "
            f"{(face.embedding ** 2).sum() ** 0.5:.4f}"
        )

    print(
        "\n=================================="
    )

    # -------------------------------------------------
    # Draw output
    # -------------------------------------------------

    output = detector.draw_detections(
        frame,
        detections,
    )

    cv2.imwrite(
        output_path,
        output,
    )

    print(
        "\nFace detection result saved to:"
    )

    print(
        output_path
    )


if __name__ == "__main__":
    main()