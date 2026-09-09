from dataclasses import dataclass
from typing import List, Tuple

from huggingface_hub import hf_hub_download
from ultralytics import YOLO

from .config import (
    PLATE_MODEL_REPO,
    PLATE_MODEL_FILE,
    PLATE_CONFIDENCE,
    PLATE_IOU,
    PLATE_IMAGE_SIZE,
)


@dataclass
class PlateDetection:
    confidence: float
    bbox: Tuple[int, int, int, int]


class LicensePlateDetector:
    """
    IBVAP license plate detection module.

    Downloads and loads a dedicated license plate
    detection model and detects plate regions.
    """

    def __init__(
        self,
        confidence: float = PLATE_CONFIDENCE,
        iou: float = PLATE_IOU,
        image_size: int = PLATE_IMAGE_SIZE,
    ):
        print("Loading IBVAP license plate model...")

        model_path = hf_hub_download(
            repo_id=PLATE_MODEL_REPO,
            filename=PLATE_MODEL_FILE,
        )

        self.model = YOLO(model_path)

        self.confidence = confidence
        self.iou = iou
        self.image_size = image_size

        print("License plate model ready.")

    def detect(self, frame) -> List[PlateDetection]:
        """
        Detect license plates in a single image/frame.
        """

        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.image_size,
            verbose=False,
        )

        detections = []

        if not results:
            return detections

        result = results[0]

        if result.boxes is None:
            return detections

        for box in result.boxes:
            confidence = float(box.conf[0].item())

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append(
                PlateDetection(
                    confidence=confidence,
                    bbox=(
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2),
                    ),
                )
            )

        return detections