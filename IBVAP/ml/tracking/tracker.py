from dataclasses import dataclass
from typing import List, Tuple

from ultralytics import YOLO

from ml.detection.config import (
    MODEL_NAME,
    CONFIDENCE_THRESHOLD,
    IOU_THRESHOLD,
    IMAGE_SIZE,
    TARGET_CLASSES,
)


@dataclass
class Track:
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]


class ObjectTracker:
    """
    IBVAP multi-object tracking module.

    Uses YOLO tracking with ByteTrack.

    Each detected person or vehicle receives
    a persistent tracking ID across video frames.
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

    def track(self, frame) -> List[Track]:
        """
        Track objects in a single video frame.

        Persistent IDs are maintained by the tracker.
        """

        results = self.model.track(
            source=frame,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.image_size,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )

        tracks = []

        if not results:
            return tracks

        result = results[0]

        if result.boxes is None:
            return tracks

        if result.boxes.id is None:
            return tracks

        track_ids = result.boxes.id.cpu().tolist()
        class_ids = result.boxes.cls.cpu().tolist()
        confidences = result.boxes.conf.cpu().tolist()
        boxes = result.boxes.xyxy.cpu().tolist()

        for track_id, class_id, confidence, bbox in zip(
            track_ids,
            class_ids,
            confidences,
            boxes,
        ):
            class_id = int(class_id)

            if class_id not in TARGET_CLASSES:
                continue

            x1, y1, x2, y2 = bbox

            tracks.append(
                Track(
                    track_id=int(track_id),
                    class_id=class_id,
                    class_name=TARGET_CLASSES[class_id],
                    confidence=float(confidence),
                    bbox=(
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2),
                    ),
                )
            )

        return tracks