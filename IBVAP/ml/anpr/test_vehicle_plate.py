import cv2

from ml.anpr.vehicle_plate import VehiclePlateAnalyzer
from ml.tracking.tracker import ObjectTracker


def main():

    image_path = "data/demo/anpr_test.jpg"
    output_path = "data/demo/complete_anpr_result.jpg"

    frame = cv2.imread(
        image_path
    )

    if frame is None:
        print(
            f"ERROR: Could not load image: "
            f"{image_path}"
        )
        return

    print("\n======================================")
    print("       IBVAP COMPLETE ANPR TEST")
    print("======================================")

    # -------------------------------------------------
    # Step 1 — Tracking
    # -------------------------------------------------

    print("\n[1/3] Loading vehicle tracker...")

    tracker = ObjectTracker()

    tracks = tracker.track(
        frame
    )

    print(
        f"Tracked objects: {len(tracks)}"
    )

    # -------------------------------------------------
    # Step 2 — Vehicle + plate + OCR
    # -------------------------------------------------

    print(
        "\n[2/3] Running complete ANPR..."
    )

    analyzer = VehiclePlateAnalyzer()

    results = analyzer.analyze(
        frame,
        tracks,
    )

    # -------------------------------------------------
    # Step 3 — Display results
    # -------------------------------------------------

    print(
        "\n[3/3] ANPR results..."
    )

    print(
        "\n========== COMPLETE ANPR RESULTS =========="
    )

    if not results:

        print(
            "No license plates detected."
        )

    for index, result in enumerate(
        results,
        start=1,
    ):

        print(
            f"\nVehicle/Plate {index}"
        )

        print(
            f"Track ID        : "
            f"{result.track_id}"
        )

        print(
            f"Vehicle         : "
            f"{result.vehicle_class}"
        )

        print(
            f"Vehicle bbox    : "
            f"{result.vehicle_bbox}"
        )

        print(
            f"Plate bbox      : "
            f"{result.plate_bbox}"
        )

        print(
            f"Plate confidence: "
            f"{result.plate_confidence:.4f}"
        )

        if result.plate_text:

            print(
                f"Plate text      : "
                f"{result.plate_text}"
            )

            print(
                f"OCR confidence  : "
                f"{result.ocr_confidence:.4f}"
            )

        else:

            print(
                "Plate text      : "
                "OCR could not read plate"
            )

        # -------------------------------------------------
        # Draw vehicle bounding box.
        # -------------------------------------------------

        vx1, vy1, vx2, vy2 = (
            result.vehicle_bbox
        )

        cv2.rectangle(
            frame,
            (vx1, vy1),
            (vx2, vy2),
            (255, 0, 0),
            2,
        )

        # -------------------------------------------------
        # Draw plate bounding box.
        # -------------------------------------------------

        px1, py1, px2, py2 = (
            result.plate_bbox
        )

        cv2.rectangle(
            frame,
            (px1, py1),
            (px2, py2),
            (0, 255, 0),
            2,
        )

        # -------------------------------------------------
        # Display plate number.
        # -------------------------------------------------

        if result.plate_text:

            label = (
                f"ID:{result.track_id} "
                f"{result.plate_text}"
            )

        else:

            label = (
                f"ID:{result.track_id} "
                "OCR FAILED"
            )

        cv2.putText(
            frame,
            label,
            (
                px1,
                max(
                    py1 - 10,
                    20,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
        )

    print(
        "\n=========================================="
    )

    # -------------------------------------------------
    # Save output.
    # -------------------------------------------------

    cv2.imwrite(
        output_path,
        frame,
    )

    print(
        "\nComplete ANPR result saved to:"
    )

    print(
        output_path
    )


if __name__ == "__main__":
    main()