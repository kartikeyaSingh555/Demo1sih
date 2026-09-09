from dataclasses import dataclass
from typing import List, Tuple

from huggingface_hub import hf_hub_download
from ultralytics import YOLO

from .config import (
    MODEL_REPO,
    MODEL_FILE,
    CONFIDENCE_THRESHOLD,
    IOU_THRESHOLD,
    IMAGE_SIZE,
    WEAPON_CLASSES,
)


@dataclass
class WeaponDetection:
    """
    Represents one detected weapon.
    """

    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]


class WeaponDetector:
    """
    IBVAP weapon detection module.

    Detects:
        - Gun
        - Knife

    This module only detects weapon-like objects.

    It does NOT determine whether a person is
    dangerous or whether an incident is a threat.
    """

    def __init__(
        self,
        confidence: float = CONFIDENCE_THRESHOLD,
        iou: float = IOU_THRESHOLD,
        image_size: int = IMAGE_SIZE,
    ):

        print(
            "\nLoading IBVAP weapon detection model..."
        )

        model_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILE,
        )

        self.model = YOLO(
            model_path
        )

        self.confidence = confidence
        self.iou = iou
        self.image_size = image_size

        print(
            "Weapon detection model ready."
        )

    def detect(
        self,
        frame,
    ) -> List[WeaponDetection]:
        """
        Detect weapons in one image/frame.
        """

        if frame is None:
            return []

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

            class_id = int(
                box.cls[0].item()
            )

            if class_id not in WEAPON_CLASSES:
                continue

            confidence = float(
                box.conf[0].item()
            )

            x1, y1, x2, y2 = (
                box.xyxy[0].tolist()
            )

            detections.append(
                WeaponDetection(
                    class_id=class_id,
                    class_name=WEAPON_CLASSES[
                        class_id
                    ],
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