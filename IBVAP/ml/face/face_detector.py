from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
from insightface.app import FaceAnalysis


@dataclass
class FaceDetection:
    """
    Represents one detected face.
    """

    bbox: Tuple[int, int, int, int]
    confidence: float
    embedding: np.ndarray


class FaceDetector:
    """
    IBVAP face detection and embedding module.

    Uses InsightFace.

    InsightFace provides:
        - Face detection
        - Face alignment
        - Face recognition embedding

    ArcFace is used by the recognition pipeline
    to generate identity embeddings.
    """

    def __init__(
        self,
        model_name: str = "buffalo_l",
        det_size: Tuple[int, int] = (640, 640),
    ):

        print(
            "\nLoading IBVAP InsightFace model..."
        )

        self.app = FaceAnalysis(
            name=model_name,
            providers=[
                "CPUExecutionProvider"
            ],
        )

        self.app.prepare(
            ctx_id=0,
            det_size=det_size,
        )

        print(
            "InsightFace model ready."
        )

    def detect(
        self,
        frame,
    ) -> List[FaceDetection]:
        """
        Detect faces and generate their embeddings.
        """

        if frame is None:
            return []

        if not isinstance(
            frame,
            np.ndarray,
        ):
            return []

        if frame.size == 0:
            return []

        faces = self.app.get(
            frame
        )

        detections = []

        for face in faces:

            x1, y1, x2, y2 = (
                face.bbox.astype(int)
            )

            confidence = float(
                face.det_score
            )

            embedding = np.asarray(
                face.embedding,
                dtype=np.float32,
            )

            detections.append(
                FaceDetection(
                    bbox=(
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2),
                    ),
                    confidence=confidence,
                    embedding=embedding,
                )
            )

        return detections

    def draw_detections(
        self,
        frame,
        detections: List[FaceDetection],
    ):
        """
        Draw detected face boxes on a frame.
        """

        output = frame.copy()

        for index, face in enumerate(
            detections,
            start=1,
        ):

            x1, y1, x2, y2 = (
                face.bbox
            )

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            label = (
                f"Face {index} "
                f"{face.confidence:.2f}"
            )

            cv2.putText(
                output,
                label,
                (
                    x1,
                    max(y1 - 10, 20),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )

        return output