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
    DRONE_CLASS_ID,
    DRONE_CLASS_NAME,
)


@dataclass
class DroneDetection:
    """
    Represents one detected drone.
    """

    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]


class DroneDetector:
    """
    IBVAP drone detection module.

    Downloads a dedicated YOLO11 drone model
    from Hugging Face and loads the local
    model weights.

    This module only reports drone detections.
    It does not determine threat level.
    """

    def __init__(
        self,
        model_repo: str = MODEL_REPO,
        model_file: str = MODEL_FILE,
        confidence: float = CONFIDENCE_THRESHOLD,
        iou: float = IOU_THRESHOLD,
        image_size: int = IMAGE_SIZE,
    ):

        print(
            "\nLoading IBVAP drone detection model..."
        )

        print(
            "Downloading/loading drone model..."
        )

        model_path = hf_hub_download(
            repo_id=model_repo,
            filename=model_file,
        )

        print(
            f"Model path: {model_path}"
        )

        self.model = YOLO(
            model_path
        )

        self.confidence = confidence
        self.iou = iou
        self.image_size = image_size

        print(
            "Drone detection model ready."
        )

    def detect(
        self,
        frame,
    ) -> List[DroneDetection]:
        """
        Detect drones in one image/frame.
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

            if class_id != DRONE_CLASS_ID:
                continue

            confidence = float(
                box.conf[0].item()
            )

            x1, y1, x2, y2 = (
                box.xyxy[0].tolist()
            )

            detections.append(
                DroneDetection(
                    class_id=class_id,
                    class_name=DRONE_CLASS_NAME,
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