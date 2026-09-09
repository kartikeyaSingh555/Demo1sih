import json

import cv2

from ml.weapon.detector import (
    WeaponDetector,
)

from ml.weapon.weapon_event import (
    WeaponEventGenerator,
)


IMAGE_PATH = (
    "data/demo/weapon_test.jpg"
)


def main():

    print("\n======================================")
    print("       IBVAP WEAPON EVENT TEST")
    print("======================================")

    # -----------------------------------------
    # STEP 1
    # -----------------------------------------

    print(
        "\n[1/3] Loading weapon detector..."
    )

    detector = WeaponDetector()

    # -----------------------------------------
    # STEP 2
    # -----------------------------------------

    print(
        "\n[2/3] Detecting weapons..."
    )

    frame = cv2.imread(
        IMAGE_PATH
    )

    if frame is None:

        print(
            f"ERROR: Could not load image: "
            f"{IMAGE_PATH}"
        )

        return

    detections = detector.detect(
        frame
    )

    print(
        f"Weapons detected: "
        f"{len(detections)}"
    )

    # -----------------------------------------
    # STEP 3
    # -----------------------------------------

    print(
        "\n[3/3] Generating IBVAP events..."
    )

    generator = WeaponEventGenerator()

    events = generator.generate_many(
        detections
    )

    print(
        "\n========== WEAPON EVENTS =========="
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
        "\n===================================="
    )

    print(
        "WEAPON EVENT GENERATION SUCCESS"
    )

    print(
        "===================================="
    )


if __name__ == "__main__":
    main()