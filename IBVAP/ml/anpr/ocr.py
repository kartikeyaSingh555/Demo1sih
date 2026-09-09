import re
from dataclasses import dataclass
from typing import Optional

import cv2
import easyocr

from .ocr_config import (
    OCR_LANGUAGES,
    OCR_GPU,
    OCR_CONFIDENCE,
    PLATE_SCALE,
    PLATE_PADDING,
    MIN_PLATE_WIDTH,
    MIN_PLATE_HEIGHT,
)


@dataclass
class OCRResult:
    text: str
    confidence: float


class LicensePlateOCR:
    """
    IBVAP license plate OCR module.

    Performs multiple preprocessing techniques and
    selects the strongest OCR result.
    """

    def __init__(
        self,
        languages=None,
        gpu=OCR_GPU,
    ):
        if languages is None:
            languages = OCR_LANGUAGES

        print("Loading IBVAP OCR model...")

        self.reader = easyocr.Reader(
            languages,
            gpu=gpu,
        )

        print("OCR model ready.")

    def preprocess_variants(self, plate_image):
        """
        Generate multiple enhanced versions of the
        license plate for OCR.
        """

        if plate_image is None or plate_image.size == 0:
            return []

        height, width = plate_image.shape[:2]

        if (
            width < MIN_PLATE_WIDTH
            or height < MIN_PLATE_HEIGHT
        ):
            return []

        # ------------------------------------------------
        # Add padding around the detected plate.
        # ------------------------------------------------

        padded = cv2.copyMakeBorder(
            plate_image,
            PLATE_PADDING,
            PLATE_PADDING,
            PLATE_PADDING,
            PLATE_PADDING,
            cv2.BORDER_REPLICATE,
        )

        # ------------------------------------------------
        # Enlarge the plate.
        # ------------------------------------------------

        scaled = cv2.resize(
            padded,
            None,
            fx=PLATE_SCALE,
            fy=PLATE_SCALE,
            interpolation=cv2.INTER_CUBIC,
        )

        # ------------------------------------------------
        # Grayscale.
        # ------------------------------------------------

        gray = cv2.cvtColor(
            scaled,
            cv2.COLOR_BGR2GRAY,
        )

        # ------------------------------------------------
        # Histogram equalization.
        # ------------------------------------------------

        equalized = cv2.equalizeHist(gray)

        # ------------------------------------------------
        # Adaptive threshold.
        # ------------------------------------------------

        threshold = cv2.adaptiveThreshold(
            equalized,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11,
        )

        # ------------------------------------------------
        # Otsu threshold.
        # ------------------------------------------------

        _, otsu = cv2.threshold(
            equalized,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

        # ------------------------------------------------
        # Sharpened grayscale.
        # ------------------------------------------------

        blurred = cv2.GaussianBlur(
            equalized,
            (0, 0),
            3,
        )

        sharpened = cv2.addWeighted(
            equalized,
            1.8,
            blurred,
            -0.8,
            0,
        )

        return [
            scaled,
            equalized,
            threshold,
            otsu,
            sharpened,
        ]

    def clean_text(self, text: str) -> str:
        """
        Keep only characters useful for vehicle
        registration numbers.
        """

        text = text.upper()

        text = re.sub(
            r"[^A-Z0-9]",
            "",
            text,
        )

        return text

    def read(self, plate_image) -> Optional[OCRResult]:
        """
        Run OCR across multiple preprocessing variants.
        """

        variants = self.preprocess_variants(
            plate_image
        )

        if not variants:
            return None

        best_text = ""
        best_confidence = 0.0

        for variant in variants:

            results = self.reader.readtext(
                variant,
                detail=1,
                paragraph=False,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                text_threshold=0.4,
                low_text=0.2,
                link_threshold=0.2,
                mag_ratio=1.0,
            )

            for _, text, confidence in results:

                confidence = float(confidence)

                cleaned = self.clean_text(
                    text
                )

                if not cleaned:
                    continue

                if confidence < OCR_CONFIDENCE:
                    continue

                if confidence > best_confidence:
                    best_text = cleaned
                    best_confidence = confidence

        if not best_text:
            return None

        return OCRResult(
            text=best_text,
            confidence=best_confidence,
        )