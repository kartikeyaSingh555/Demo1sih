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
    "data/demo/face_test.jpg"
)


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
    print("    IBVAP DATABASE FACE RECOGNITION")
    print("======================================")

    print(
        "\n[1/4] Loading face database..."
    )

    database = load_database()

    if database is None:
        return

    print(
        f"Enrolled identities: "
        f"{len(database)}"
    )

    print(
        "\nEnrolled personnel:"
    )

    for identity_id, record in (
        database.items()
    ):

        print(
            f"  {identity_id} "
            f"-> {record['name']}"
        )

    print(
        "\n[2/4] Loading InsightFace..."
    )

    detector = FaceDetector()

    print(
        "\n[3/4] Detecting face..."
    )

    frame = cv2.imread(
        IMAGE_PATH
    )

    if frame is None:

        print(
            f"ERROR: Could not load: "
            f"{IMAGE_PATH}"
        )

        return

    detections = detector.detect(
        frame
    )

    print(
        f"Faces detected: "
        f"{len(detections)}"
    )

    if not detections:

        print(
            "No faces found."
        )

        return

    print(
        "\n[4/4] Matching against database..."
    )

    matcher = FaceMatcher(
        threshold=0.45
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
        "\n========== RECOGNITION RESULTS =========="
    )

    for index, face in enumerate(
        detections,
        start=1,
    ):

        result = matcher.match(
            face.embedding
        )

        print(
            f"\nFace {index}"
        )

        print(
            f"Detection confidence : "
            f"{face.confidence:.4f}"
        )

        if result is None:

            print(
                "Recognition          : "
                "UNKNOWN"
            )

            continue

        print(
            f"Similarity            : "
            f"{result['similarity']:.4f}"
        )

        if result["matched"]:

            print(
                "Recognition          : "
                "AUTHORIZED"
            )

            print(
                f"Identity ID          : "
                f"{result['identity_id']}"
            )

            print(
                f"Name                 : "
                f"{result['name']}"
            )

        else:

            print(
                "Recognition          : "
                "UNKNOWN"
            )

    print(
        "\n=========================================="
    )


if __name__ == "__main__":
    main()