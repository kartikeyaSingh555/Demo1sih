import cv2
import numpy as np

from ml.face.face_detector import FaceDetector
from ml.face.face_matcher import FaceMatcher


def main():

    image_path = "data/demo/face_test.jpg"

    print("\n======================================")
    print("       IBVAP FACE RECOGNITION TEST")
    print("======================================")

    frame = cv2.imread(image_path)

    if frame is None:
        print(
            f"ERROR: Could not load image: "
            f"{image_path}"
        )
        return

    print(
        "\n[1/3] Loading InsightFace..."
    )

    detector = FaceDetector()

    print(
        "\n[2/3] Detecting face..."
    )

    detections = detector.detect(frame)

    if not detections:
        print(
            "ERROR: No face detected."
        )
        return

    print(
        f"Faces detected: "
        f"{len(detections)}"
    )

    face = detections[0]

    print(
        "\nEmbedding generated:"
    )

    print(
        f"Shape: {face.embedding.shape}"
    )

    print(
        f"Dimension: "
        f"{len(face.embedding)}"
    )

    print(
        "\n[3/3] Testing ArcFace matching..."
    )

    matcher = FaceMatcher(
        threshold=0.45
    )

    # ------------------------------------------------
    # DEMO ENROLLMENT
    # ------------------------------------------------
    #
    # In the real IBVAP system this embedding will
    # come from the authorized personnel database.
    #
    # For this test we temporarily enroll the
    # detected face itself.
    #

    matcher.add_identity(
        identity_id="PERSON_001",
        name="Demo Authorized Person",
        embedding=face.embedding,
    )

    print(
        "\nAuthorized identity enrolled:"
    )

    print(
        "ID   : PERSON_001"
    )

    print(
        "Name : Demo Authorized Person"
    )

    # ------------------------------------------------
    # MATCH THE SAME FACE
    # ------------------------------------------------

    result = matcher.match(
        face.embedding
    )

    print(
        "\n========== MATCH RESULT =========="
    )

    if result is None:

        print(
            "No match result."
        )

    else:

        print(
            f"Matched       : "
            f"{result['matched']}"
        )

        print(
            f"Identity ID   : "
            f"{result['identity_id']}"
        )

        print(
            f"Name          : "
            f"{result['name']}"
        )

        print(
            f"Similarity     : "
            f"{result['similarity']:.4f}"
        )

    print(
        "\n=================================="
    )


if __name__ == "__main__":
    main()