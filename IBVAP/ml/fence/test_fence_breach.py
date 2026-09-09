import cv2
import json

from ml.fence.fence_detector import FenceDetector
from ml.fence.breach_analyzer import FenceBreachAnalyzer
from ml.fence.fence_event import FenceEventGenerator


NORMAL_IMAGE = "data/demo/fence_normal.jpg"
DAMAGED_IMAGE = "data/demo/fence_damaged_test.jpg"

OUTPUT_IMAGE = "data/demo/fence_damage_result.jpg"


def main():

    print("\n======================================")
    print("      IBVAP PHYSICAL FENCE TEST")
    print("======================================")

    # --------------------------------------------------
    # 1. Load normal fence image
    # --------------------------------------------------

    print("\n[1/5] Loading normal fence image...")

    normal_frame = cv2.imread(NORMAL_IMAGE)

    if normal_frame is None:

        print(
            f"ERROR: Could not load: {NORMAL_IMAGE}"
        )

        return

    print(
        f"Normal image loaded: {NORMAL_IMAGE}"
    )

    # --------------------------------------------------
    # 2. Create fence detector
    # --------------------------------------------------

    print("\n[2/5] Establishing normal fence reference...")

    detector = FenceDetector()

    detector.set_reference(
        normal_frame
    )

    print(
        "Normal fence reference established."
    )

    # --------------------------------------------------
    # 3. Load damaged image
    # --------------------------------------------------

    print("\n[3/5] Loading damaged fence image...")

    damaged_frame = cv2.imread(DAMAGED_IMAGE)

    if damaged_frame is None:

        print(
            f"ERROR: Could not load: "
            f"{DAMAGED_IMAGE}"
        )

        return

    print(
        f"Damaged image loaded: "
        f"{DAMAGED_IMAGE}"
    )

    # --------------------------------------------------
    # 4. Detect fence condition
    # --------------------------------------------------

    print("\n[4/5] Analyzing fence condition...")

    condition = detector.detect(
        damaged_frame
    )

    print(
        "\n========== FENCE CONDITION =========="
    )

    print(
        f"Abnormal       : "
        f"{condition.abnormal}"
    )

    print(
        f"Change score   : "
        f"{condition.change_score:.4f}"
    )

    print(
        f"Change bbox    : "
        f"{condition.bbox}"
    )

    print(
        "====================================="
    )

    # --------------------------------------------------
    # Draw detected change
    # --------------------------------------------------

    output = damaged_frame.copy()

    if condition.bbox is not None:

        x1, y1, x2, y2 = (
            condition.bbox
        )

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            3,
        )

        cv2.putText(
            output,
            "FENCE CHANGE DETECTED",
            (x1, max(y1 - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
        )

    # --------------------------------------------------
    # 5. Temporal confirmation
    # --------------------------------------------------

    print(
        "\n[5/5] Running temporal confirmation..."
    )

    analyzer = FenceBreachAnalyzer(
        confirmation_frames=1,
        confidence_threshold=0.60,
    )

    # Convert visual change score into a
    # normalized confidence for this prototype.

    confidence = min(
        condition.change_score / 0.50,
        1.0,
    )

    result = analyzer.update(
        abnormal=condition.abnormal,
        confidence=confidence,
    )

    print(
        f"\nConfirmed       : "
        f"{result.confirmed}"
    )

    print(
        f"Confidence      : "
        f"{result.confidence:.4f}"
    )

    print(
        f"Frames          : "
        f"{result.consecutive_frames}"
    )

    # --------------------------------------------------
    # Generate event
    # --------------------------------------------------

    if result.confirmed:

        generator = FenceEventGenerator()

        event = generator.generate(
            breach_type="CUT",
            confidence=result.confidence,
            severity="CRITICAL",
            camera_id="CAM_BORDER_01",
            bbox=condition.bbox,
            reason=(
                "Physical fence condition changed "
                "from the established reference."
            ),
        )

        print(
            "\n========== FENCE EVENT =========="
        )

        print(
            json.dumps(
                event.to_dict(),
                indent=4,
            )
        )

        print(
            "================================="
        )

        cv2.putText(
            output,
            "ALARM: FENCE DAMAGE",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            3,
        )

    else:

        print(
            "\nNo confirmed fence damage."
        )

    cv2.imwrite(
        OUTPUT_IMAGE,
        output,
    )

    print(
        f"\nResult image saved to:"
    )

    print(
        OUTPUT_IMAGE
    )

    print(
        "\n======================================"
    )


if __name__ == "__main__":
    main()