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
    "data/demo/unknown_face_real.jpg"
)


def load_database():

    path = Path(DATABASE_PATH)

    if not path.exists():

        print(
            "ERROR: Face database not found:"
        )

        print(DATABASE_PATH)

        return None

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


def main():

    print("\n======================================")
    print("       IBVAP UNKNOWN FACE TEST")
    print("======================================")

    print(
        "\n[1/4] Loading enrolled database..."
    )

    database = load_database()

    if database is None:
        return

    print(
        f"Enrolled identities: "
        f"{len(database)}"
    )

    print(
        "\n[2/4] Loading InsightFace..."
    )

    detector = FaceDetector()

    print(
        "\n[3/4] Loading unknown face image..."
    )

    frame = cv2.imread(
        IMAGE_PATH
    )

    if frame is None:

        print(
            f"ERROR: Could not load image:"
        )

        print(
            IMAGE_PATH
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
            "ERROR: No face detected."
        )

        return

    print(
        "\n[4/4] Matching against enrolled "
        "personnel..."
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
        "\n========== UNKNOWN FACE RESULT =========="
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
                "Recognition          : UNKNOWN"
            )

            continue

        print(
            f"Best similarity       : "
            f"{result['similarity']:.4f}"
        )

        if result["matched"]:

            print(
                "Recognition          : "
                "AUTHORIZED"
            )

            print(
                f"Identity ID           : "
                f"{result['identity_id']}"
            )

            print(
                f"Name                  : "
                f"{result['name']}"
            )

        else:

            print(
                "Recognition          : "
                "UNKNOWN"
            )

            print(
                "Reason                : "
                "Similarity below threshold"
            )

    print(
        "\n=========================================="
    )


if __name__ == "__main__":
    main()