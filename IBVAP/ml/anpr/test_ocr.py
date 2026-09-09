import cv2

from ml.anpr.ocr import LicensePlateOCR
from ml.anpr.vehicle_plate import VehiclePlateAnalyzer
from ml.tracking.tracker import ObjectTracker


def main():

    image_path = "data/demo/anpr_test.jpg"
    output_path = "data/demo/anpr_ocr_result.jpg"

    frame = cv2.imread(image_path)

    if frame is None:
        print(f"ERROR: Could not load image: {image_path}")
        return

    print("\n======================================")
    print("       IBVAP ANPR PIPELINE TEST")
    print("======================================")

    # -------------------------------------------------
    # STEP 1: TRACK VEHICLES
    # -------------------------------------------------

    print("\n[1/3] Loading vehicle tracker...")

    tracker = ObjectTracker()
    tracks = tracker.track(frame)

    print(f"Tracked objects: {len(tracks)}")

    # -------------------------------------------------
    # STEP 2: DETECT PLATES
    # -------------------------------------------------

    print("\n[2/3] Detecting license plates...")

    analyzer = VehiclePlateAnalyzer()
    plate_results = analyzer.analyze(frame, tracks)

    print(f"License plates detected: {len(plate_results)}")

    # -------------------------------------------------
    # STEP 3: OCR
    # -------------------------------------------------

    print("\n[3/3] Running OCR...")

    ocr = LicensePlateOCR()

    print("\n========== ANPR RESULTS ==========")

    if not plate_results:
        print("No license plates detected.")

    for index, result in enumerate(plate_results, start=1):

        px1, py1, px2, py2 = result.plate_bbox

        # ---------------------------------------------
        # Expand detected plate region and enforce bounds
        # ---------------------------------------------
        padding_x = 8
        padding_y = 5

        px1 = max(0, px1 - padding_x)
        py1 = max(0, py1 - padding_y)
        px2 = min(frame.shape[1], px2 + padding_x)
        py2 = min(frame.shape[0], py2 + padding_y)

        # ---------------------------------------------
        # Crop plate
        # ---------------------------------------------
        plate_crop = frame[py1:py2, px1:px2]

        # Save enlarged plate crop for visual inspection
        if plate_crop.size > 0:
            enlarged_plate = cv2.resize(
                plate_crop,
                None,
                fx=10,
                fy=10,
                interpolation=cv2.INTER_CUBIC,
            )
            cv2.imwrite(
                f"data/demo/plate_crop_{index}_enlarged.jpg",
                enlarged_plate,
            )

        # ---------------------------------------------
        # Save padded raw plate crop for inspection
        # ---------------------------------------------
        crop_path = f"data/demo/plate_crop_{index}.jpg"
        cv2.imwrite(crop_path, plate_crop)

        # ---------------------------------------------
        # OCR
        # ---------------------------------------------
        ocr_result = ocr.read(plate_crop)

        print(f"\nPlate {index}")
        print(f"Track ID        : {result.track_id}")
        print(f"Vehicle         : {result.vehicle_class}")
        print(f"Plate confidence: {result.plate_confidence:.2f}")
        print(f"Plate crop      : {crop_path}")

        if ocr_result:
            print(f"Plate text      : {ocr_result.text}")
            print(f"OCR confidence  : {ocr_result.confidence:.2f}")
            label = f"ID:{result.track_id} {ocr_result.text}"
        else:
            print("Plate text      : OCR could not read plate")
            label = f"ID:{result.track_id} OCR failed"

        # ---------------------------------------------
        # Draw plate bounding box and label on frame
        # ---------------------------------------------
        cv2.rectangle(
            frame,
            (px1, py1),
            (px2, py2),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            label,
            (px1, max(py1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
        )

    print("\n=================================")

    cv2.imwrite(output_path, frame)

    print("\nANPR result image saved to:")
    print(output_path)


if __name__ == "__main__":
    main()