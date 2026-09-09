import json

import cv2
import numpy as np

from ml.face.face_event import (
    FaceEventGenerator,
)

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
    print("       IBVAP FACE EVENT TEST")
    print("======================================")

    # ------------------------------------------------
    # STEP 1
    # ------------------------------------------------

    print(
        "\n[1/6] Loading face database..."
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

    # ------------------------------------------------
    # STEP 2
    # ------------------------------------------------

    print(
        "\n[2/6] Loading person tracker..."
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

    # ------------------------------------------------
    # STEP 3
    # ------------------------------------------------

    print(
        "\n[3/6] Loading face matcher..."
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

    # ------------------------------------------------
    # STEP 4
    # ------------------------------------------------

    print(
        "\n[4/6] Running face recognition..."
    )

    recognizer = (
        FaceTrackingRecognizer(
            matcher=matcher
        )
    )

    results = recognizer.recognize(
        frame,
        tracks,
    )

    print(
        f"Faces recognized: "
        f"{len(results)}"
    )

    # ------------------------------------------------
    # STEP 5
    # ------------------------------------------------

    print(
        "\n[5/6] Generating ML events..."
    )

    generator = FaceEventGenerator()

    events = generator.generate_many(
        results
    )

    # ------------------------------------------------
    # STEP 6
    # ------------------------------------------------

    print(
        "\n[6/6] IBVAP FACE EVENTS"
    )

    print(
        "======================================"
    )

    for index, event in enumerate(
        events,
        start=1,
    ):

        print(
            f"\nEvent {index}"
        )

        print(
            json.dumps(
                event.to_dict(),
                indent=4,
            )
        )

    print(
        "\n======================================"
    )

    print(
        "FACE EVENT GENERATION SUCCESS"
    )

    print(
        "======================================"
    )


if __name__ == "__main__":
    main()