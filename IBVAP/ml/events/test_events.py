import json

from ml.events import (
    MLEventBuilder,
)


def print_event(
    title,
    event,
):

    print(
        f"\n========== {title} =========="
    )

    print(
        json.dumps(
            event.to_dict(),
            indent=4,
        )
    )

    print(
        "================================"
    )


def main():

    print("\n======================================")
    print("       IBVAP UNIFIED ML EVENTS")
    print("======================================")

    builder = MLEventBuilder()

    # ------------------------------------------
    # Weapon
    # ------------------------------------------

    weapon_event = builder.from_weapon(
        camera_id="CAM_BORDER_01",
        weapon_type="knife",
        confidence=0.91,
        bbox=(100, 150, 200, 300),
    )

    print_event(
        "WEAPON EVENT",
        weapon_event,
    )

    # ------------------------------------------
    # Drone
    # ------------------------------------------

    drone_event = builder.from_drone(
        camera_id="CAM_BORDER_02",
        confidence=0.87,
        bbox=(300, 100, 380, 170),
    )

    print_event(
        "DRONE EVENT",
        drone_event,
    )

    # ------------------------------------------
    # Face
    # ------------------------------------------

    face_event = builder.from_face(
        camera_id="CAM_BORDER_01",
        track_id=17,
        identity_id=None,
        name=None,
        status="UNKNOWN",
        face_confidence=0.89,
        match_confidence=0.21,
        bbox=(200, 100, 300, 250),
    )

    print_event(
        "FACE EVENT",
        face_event,
    )

    # ------------------------------------------
    # Fence
    # ------------------------------------------

    fence_event = builder.from_fence(
        camera_id="CAM_BORDER_03",
        breach_type="CUT",
        confidence=0.91,
        severity="CRITICAL",
    )

    print_event(
        "FENCE EVENT",
        fence_event,
    )

    # ------------------------------------------
    # Camera health
    # ------------------------------------------

    camera_event = builder.from_camera_health(
        camera_id="CAM_BORDER_04",
        status="OFFLINE",
        brightness=0.0,
        sharpness=0.0,
        black_ratio=1.0,
        issues=[
            "BLACK_SCREEN",
            "TOO_DARK",
        ],
        severity="CRITICAL",
    )

    print_event(
        "CAMERA HEALTH EVENT",
        camera_event,
    )

    print(
        "\n======================================"
    )

    print(
        "UNIFIED ML EVENT PIPELINE TEST SUCCESS"
    )

    print(
        "======================================"
    )


if __name__ == "__main__":
    main()