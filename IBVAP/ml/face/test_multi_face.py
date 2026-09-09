import json
from pathlib import Path

import cv2
import numpy as np

from ml.face.face_detector import FaceDetector
from ml.face.face_matcher import FaceMatcher


DATABASE_PATH = (
    "data/watchlist/embeddings/"
    "face_database.json"
)

IMAGE_PATH = (
    "data/demo/multi_face_test.jpg"
)

THRESHOLD = 0.45


def load_database():

    path = Path(
        DATABASE_PATH
    )

    if not path.exists():

        print(
            "\nERROR: Face database not found:"
        )

        print(
            DATABASE_PATH
        )

        return None

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


def main():

    print("\n======================================")
    print("       IBVAP MULTI-FACE TEST")
    print("======================================")

    print(
        "\n[1/5] Loading face database..."
    )

    database = load_database()

    if database is None:
        return

    print(
        f"Enrolled identities: "
        f"{len(database)}"
    )

    print(
        "\n[2/5] Loading InsightFace..."
    )

    detector = FaceDetector()

    print(
        "\n[3/5] Loading test image..."
    )

    frame = cv2.imread(
        IMAGE_PATH
    )

    if frame is None:

        print(
            "\nERROR: Could not load:"
        )

        print(
            IMAGE_PATH
        )

        return

    print(
        f"Image size: "
        f"{frame.shape[1]}x"
        f"{frame.shape[0]}"
    )

    print(
        "\n[4/5] Detecting faces..."
    )

    detections = detector.detect(
        frame
    )

    print(
        f"Faces detected: "
        f"{len(detections)}"
    )

    if not detections:

        print(
            "ERROR: No faces detected."
        )

        return

    print(
        "\n[5/5] Recognizing faces..."
    )

    matcher = FaceMatcher(
        threshold=THRESHOLD
    )

    for identity_id, record in (
        database.items()
    ):

        embedding = np.asarray(
            record["embedding"],
            dtype=np.float32,
        )

        matcher.load_identity(
            identity_id=identity_id,
            name=record["name"],
            embedding=embedding,
        )

    print(
        "\n========== MULTI-FACE RESULTS =========="
    )

    output = frame.copy()

    authorized_count = 0
    unknown_count = 0

    for index, face in enumerate(
        detections,
        start=1,
    ):

        result = matcher.match(
            face.embedding
        )

        x1, y1, x2, y2 = face.bbox

        if (
            result is not None
            and result["matched"]
        ):

            status = "AUTHORIZED"

            identity_id = (
                result["identity_id"]
            )

            name = result["name"]

            similarity = (
                result["similarity"]
            )

            authorized_count += 1

            label = (
                f"{name} "
                f"{similarity:.2f}"
            )

        else:

            status = "UNKNOWN"

            identity_id = None
            name = None

            if result is not None:
                similarity = result["similarity"]
            else:
                similarity = 0.0

            unknown_count += 1

            label = (
                f"UNKNOWN "
                f"{similarity:.2f}"
            )

        print(
            f"\nFace {index}"
        )

        print(
            f"Bounding box         : "
            f"{face.bbox}"
        )

        print(
            f"Detection confidence : "
            f"{face.confidence:.4f}"
        )

        print(
            f"Recognition           : "
            f"{status}"
        )

        print(
            f"Similarity            : "
            f"{similarity:.4f}"
        )

        if identity_id:

            print(
                f"Identity ID           : "
                f"{identity_id}"
            )

            print(
                f"Name                  : "
                f"{name}"
            )

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            output,
            label,
            (
                x1,
                max(y1 - 10, 20),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
        )

    output_path = (
        "data/demo/"
        "multi_face_result.jpg"
    )

    cv2.imwrite(
        output_path,
        output,
    )

    print(
        "\n=========================================="
    )

    print(
        f"Authorized faces : "
        f"{authorized_count}"
    )

    print(
        f"Unknown faces    : "
        f"{unknown_count}"
    )

    print(
        "\nResult image saved to:"
    )

    print(
        output_path
    )

    print(
        "\n=========================================="
    )


if __name__ == "__main__":
    main()