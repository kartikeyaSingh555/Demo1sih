from ml.pipeline.video_source import (
    VideoSource,
)

from ml.pipeline.processor import (
    IBVAPVideoProcessor,
)


VIDEO_PATH = "data/demo/test2.mp4"

CAMERA_ID = "CAM_BORDER_01"

MAX_FRAMES = 30


def main():

    print("\n======================================")
    print("       IBVAP VIDEO ML PIPELINE")
    print("======================================")

    # --------------------------------------
    # Video source
    # --------------------------------------

    print("\n[1/3] Opening video source...")

    source = VideoSource(
        VIDEO_PATH
    )

    source.open()

    print(
        f"Video source opened: "
        f"{VIDEO_PATH}"
    )

    print(
        f"Resolution: "
        f"{source.get_width()}x"
        f"{source.get_height()}"
    )

    print(
        f"FPS: "
        f"{source.get_fps():.2f}"
    )

    # --------------------------------------
    # Processor
    # --------------------------------------

    print(
        "\n[2/3] Loading IBVAP ML processor..."
    )

    processor = (
        IBVAPVideoProcessor(
            camera_id=CAMERA_ID
        )
    )

    # --------------------------------------
    # Process frames
    # --------------------------------------

    print(
        "\n[3/3] Processing video frames..."
    )

    total_frames = 0
    total_events = 0

    try:

        while total_frames < MAX_FRAMES:

            success, frame = source.read()

            if not success:
                break

            total_frames += 1

            events = (
                processor.process_frame(
                    frame
                )
            )

            total_events += len(events)

            print(
                f"Frame {total_frames:03d} "
                f"-> "
                f"{len(events)} events"
            )

    finally:

        source.release()

    print(
        "\n======================================"
    )

    print(
        f"Frames processed : "
        f"{total_frames}"
    )

    print(
        f"Events generated : "
        f"{total_events}"
    )

    print(
        "======================================"
    )

    print(
        "\nIBVAP VIDEO PIPELINE TEST COMPLETE"
    )


if __name__ == "__main__":
    main()