import json

import cv2
import numpy as np

from ml.face.face_matcher import (
    FaceMatcher,
)
from ml.face.face_tracking import (
    FaceTrackingRecognizer,
)
from ml.tracking.tracker import (
    ObjectTracker,
)


DATABASE_PATH = (
    "data/watchlist/embeddings/"
    "face_database.json"
)

IMAGE_PATH = (
    "data/demo/multi_face_test.jpg"
)


def main():

    print("\n======================================")
    print("     IBVAP FACE + TRACK TEST")
    print("======================================")

    print(
        "\n[1/5] Loading face database..."
    )

    with open(
        DATABASE_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        database = json.load(file)

    print(
        f"Enrolled identities: "
        f"{len(database)}"
    )

    print(
        "\n[2/5] Loading person tracker..."
    )

    tracker = ObjectTracker()

    frame = cv2.imread(
        IMAGE_PATH
    )

    if frame is None:

        print(
            f"ERROR: Could not load:"
        )

        print(
            IMAGE_PATH
        )

        return

    tracks = tracker.track(
        frame
    )

    person_tracks = [
        track
        for track in tracks
        if track.class_name == "person"
    ]

    print(
        f"Person tracks: "
        f"{len(person_tracks)}"
    )

    for track in person_tracks:

        print(
            f"  Track ID={track.track_id} "
            f"bbox={track.bbox}"
        )

    print(
        "\n[3/5] Loading face matcher..."
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
        "\n[4/5] Starting face-track recognizer..."
    )

    recognizer = (
        FaceTrackingRecognizer(
            matcher=matcher
        )
    )

    print(
        "\n[5/5] Recognizing faces..."
    )

    results = recognizer.recognize(
        frame,
        tracks,
    )

    print(
        "\n========== FACE TRACK RESULTS =========="
    )

    if not results:

        print(
            "No face could be associated "
            "with a person track."
        )

    for index, result in enumerate(
        results,
        start=1,
    ):

        print(
            f"\nFace {index}"
        )

        print(
            f"Track ID             : "
            f"{result.track_id}"
        )

        print(
            f"Face bbox            : "
            f"{result.face_bbox}"
        )

        print(
            f"Face confidence      : "
            f"{result.face_confidence:.4f}"
        )

        print(
            f"Recognition           : "
            f"{result.recognition_status}"
        )

        print(
            f"Similarity            : "
            f"{result.similarity:.4f}"
        )

        if result.identity_id:

            print(
                f"Identity ID           : "
                f"{result.identity_id}"
            )

            print(
                f"Name                  : "
                f"{result.name}"
            )

    print(
        "\n=========================================="
    )


if __name__ == "__main__":
    main()