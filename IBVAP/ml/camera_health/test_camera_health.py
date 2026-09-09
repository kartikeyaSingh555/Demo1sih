import json

import cv2
import numpy as np

from ml.camera_health.monitor import (
    CameraHealthMonitor,
)

from ml.camera_health.camera_event import (
    CameraHealthEventGenerator,
)


IMAGE_PATH = "data/demo/test.jpg"


def print_result(
    title,
    health,
):

    print(
        f"\n========== {title} =========="
    )

    print(
        f"Status       : {health.status}"
    )

    print(
        f"Brightness   : "
        f"{health.brightness:.2f}"
    )

    print(
        f"Sharpness    : "
        f"{health.sharpness:.2f}"
    )

    print(
        f"Black ratio  : "
        f"{health.black_ratio:.4f}"
    )

    print(
        f"Issues       : "
        f"{health.issues}"
    )

    print(
        "================================"
    )


def main():

    print("\n======================================")
    print("       IBVAP CAMERA HEALTH TEST")
    print("======================================")

    monitor = CameraHealthMonitor()

    # ------------------------------------------
    # Test 1: Normal image
    # ------------------------------------------

    print(
        "\n[1/4] Testing normal CCTV frame..."
    )

    frame = cv2.imread(
        IMAGE_PATH
    )

    if frame is None:

        print(
            f"ERROR: Could not load "
            f"{IMAGE_PATH}"
        )

        return

    health = monitor.analyze(
        frame
    )

    print_result(
        "NORMAL FRAME",
        health,
    )

    # ------------------------------------------
    # Test 2: Black frame
    # ------------------------------------------

    print(
        "\n[2/4] Testing black camera frame..."
    )

    black_frame = np.zeros_like(
        frame
    )

    black_health = monitor.analyze(
        black_frame
    )

    print_result(
        "BLACK FRAME",
        black_health,
    )

    # ------------------------------------------
    # Test 3: Blurred frame
    # ------------------------------------------

    print(
        "\n[3/4] Testing blurred camera frame..."
    )

    blurred_frame = cv2.GaussianBlur(
        frame,
        (51, 51),
        0,
    )

    blurred_health = monitor.analyze(
        blurred_frame
    )

    print_result(
        "BLURRED FRAME",
        blurred_health,
    )

    # ------------------------------------------
    # Test 4: Event generation
    # ------------------------------------------

    print(
        "\n[4/4] Generating camera health event..."
    )

    generator = (
        CameraHealthEventGenerator()
    )

    event = generator.generate(
        black_health,
        camera_id="CAM_BORDER_01",
    )

    if event is not None:

        print(
            "\n========== CAMERA EVENT =========="
        )

        print(
            json.dumps(
                event.to_dict(),
                indent=4,
            )
        )

        print(
            "=================================="
        )

    else:

        print(
            "\nNo alert event generated."
        )

    print(
        "\n======================================"
    )

    print(
        "CAMERA HEALTH TEST COMPLETE"
    )

    print(
        "======================================"
    )


if __name__ == "__main__":
    main()