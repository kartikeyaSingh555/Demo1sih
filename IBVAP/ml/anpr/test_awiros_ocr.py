import cv2

from ml.anpr.awiros_ocr import AwirosLicensePlateOCR


def main():

    image_path = "data/demo/plate_crop_1.jpg"

    print("\n======================================")
    print("       IBVAP AWIROS ANPR OCR TEST")
    print("======================================")

    print(f"\nInput plate crop:")
    print(image_path)

    plate_image = cv2.imread(
        image_path
    )

    if plate_image is None:
        print(
            f"\nERROR: Could not load image:"
            f" {image_path}"
        )
        return

    print(
        f"Crop size: "
        f"{plate_image.shape[1]}x"
        f"{plate_image.shape[0]}"
    )

    print("\nLoading Awiros OCR...")

    ocr = AwirosLicensePlateOCR(
        device="cpu"
    )

    print("\nRunning OCR...")

    result = ocr.read(
        plate_image
    )

    print("\n========== AWIROS OCR RESULT ==========")

    if result is None:

        print(
            "Plate text      : OCR could not read plate"
        )

    else:

        print(
            f"Plate text      : {result['text']}"
        )

        print(
            f"OCR confidence  : "
            f"{result['confidence']:.4f}"
        )

    print("========================================")


if __name__ == "__main__":
    main()