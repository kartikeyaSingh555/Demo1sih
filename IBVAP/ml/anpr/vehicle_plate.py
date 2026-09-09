from dataclasses import dataclass
from typing import List, Optional, Tuple

from ml.anpr.awiros_ocr import AwirosLicensePlateOCR
from ml.anpr.plate_detector import LicensePlateDetector
from ml.tracking.tracker import Track


@dataclass
class VehiclePlateDetection:
    """
    Complete vehicle + license plate + OCR result.
    """

    track_id: int
    vehicle_class: str
    vehicle_bbox: Tuple[int, int, int, int]

    plate_confidence: float
    plate_bbox: Tuple[int, int, int, int]

    plate_text: Optional[str]
    ocr_confidence: Optional[float]


class VehiclePlateAnalyzer:
    """
    IBVAP vehicle-to-license-plate ANPR analyzer.

    Pipeline:

        Tracked vehicle
              ↓
        Vehicle crop
              ↓
        Plate detection
              ↓
        Plate crop
              ↓
        Awiros Indian ANPR OCR
              ↓
        Complete ANPR result
    """

    VEHICLE_CLASSES = {
        "car",
        "motorcycle",
        "bus",
        "truck",
    }

    def __init__(self):

        print("Loading license plate detector...")

        self.plate_detector = LicensePlateDetector()

        print("Loading Awiros ANPR OCR...")

        self.ocr = AwirosLicensePlateOCR(
            device="cpu"
        )

        print("Vehicle-to-plate ANPR analyzer ready.")

    def analyze(
        self,
        frame,
        tracks: List[Track],
    ) -> List[VehiclePlateDetection]:
        """
        Analyze tracked vehicles and extract license
        plate numbers using Awiros OCR.
        """

        results = []

        frame_height, frame_width = frame.shape[:2]

        for track in tracks:

            # -------------------------------------------------
            # Only process vehicles.
            # -------------------------------------------------

            if track.class_name not in self.VEHICLE_CLASSES:
                continue

            vx1, vy1, vx2, vy2 = track.bbox

            # -------------------------------------------------
            # Keep vehicle coordinates inside frame.
            # -------------------------------------------------

            vx1 = max(0, vx1)
            vy1 = max(0, vy1)

            vx2 = min(frame_width, vx2)
            vy2 = min(frame_height, vy2)

            if vx2 <= vx1 or vy2 <= vy1:
                continue

            # -------------------------------------------------
            # Crop vehicle.
            # -------------------------------------------------

            vehicle_crop = frame[
                vy1:vy2,
                vx1:vx2,
            ]

            if vehicle_crop.size == 0:
                continue

            # -------------------------------------------------
            # Detect plates inside vehicle.
            # -------------------------------------------------

            plates = self.plate_detector.detect(
                vehicle_crop
            )

            for plate in plates:

                px1, py1, px2, py2 = plate.bbox

                # -------------------------------------------------
                # Convert plate coordinates from vehicle crop
                # back to original frame coordinates.
                # -------------------------------------------------

                original_px1 = vx1 + px1
                original_py1 = vy1 + py1
                original_px2 = vx1 + px2
                original_py2 = vy1 + py2

                # -------------------------------------------------
                # Keep plate coordinates inside frame.
                # -------------------------------------------------

                original_px1 = max(
                    0,
                    original_px1,
                )

                original_py1 = max(
                    0,
                    original_py1,
                )

                original_px2 = min(
                    frame_width,
                    original_px2,
                )

                original_py2 = min(
                    frame_height,
                    original_py2,
                )

                if (
                    original_px2 <= original_px1
                    or original_py2 <= original_py1
                ):
                    continue

                # -------------------------------------------------
                # Crop license plate from original frame.
                # -------------------------------------------------

                plate_crop = frame[
                    original_py1:original_py2,
                    original_px1:original_px2,
                ]

                if plate_crop.size == 0:
                    continue

                # -------------------------------------------------
                # Run Awiros Indian ANPR OCR.
                # -------------------------------------------------

                ocr_result = self.ocr.read(
                    plate_crop
                )

                plate_text = None
                ocr_confidence = None

                if ocr_result:

                    plate_text = ocr_result[
                        "text"
                    ]

                    ocr_confidence = float(
                        ocr_result[
                            "confidence"
                        ]
                    )

                # -------------------------------------------------
                # Store complete result.
                # -------------------------------------------------

                results.append(
                    VehiclePlateDetection(
                        track_id=track.track_id,
                        vehicle_class=track.class_name,
                        vehicle_bbox=track.bbox,
                        plate_confidence=plate.confidence,
                        plate_bbox=(
                            original_px1,
                            original_py1,
                            original_px2,
                            original_py2,
                        ),
                        plate_text=plate_text,
                        ocr_confidence=ocr_confidence,
                    )
                )

        return results