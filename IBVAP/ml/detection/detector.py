from dataclasses import dataclass
from typing import List, Tuple

from ultralytics import YOLO

from .config import (
    MODEL_NAME,
    CONFIDENCE_THRESHOLD,
    IOU_THRESHOLD,
    IMAGE_SIZE,
    TARGET_CLASSES,
)


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]


class ObjectDetector:
    """
    IBVAP object detection module.

    Detects:
    - Person
    - Car
    - Motorcycle
    - Bus
    - Truck
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        confidence: float = CONFIDENCE_THRESHOLD,
        iou: float = IOU_THRESHOLD,
        image_size: int = IMAGE_SIZE,
    ):
        self.model = YOLO(model_name)

        self.confidence = confidence
        self.iou = iou
        self.image_size = image_size

    def detect(self, frame) -> List[Detection]:
        """
        Run object detection on a single video frame.
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
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())

            if class_id not in TARGET_CLASSES:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detection = Detection(
                class_id=class_id,
                class_name=TARGET_CLASSES[class_id],
                confidence=confidence,
                bbox=(
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2),
                ),
            )

            detections.append(detection)

        return detections