from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from ml.face.face_detector import (
    FaceDetection,
    FaceDetector,
)
from ml.face.face_matcher import (
    FaceMatcher,
)
from ml.tracking.tracker import (
    Track,
)


@dataclass
class FaceTrackResult:
    """
    Associates a detected face with a person
    tracking ID and recognition result.
    """

    track_id: int

    face_bbox: Tuple[int, int, int, int]

    face_confidence: float

    identity_id: Optional[str]

    name: Optional[str]

    similarity: float

    recognition_status: str


class FaceTrackingRecognizer:
    """
    IBVAP face + person tracking integration.

    Pipeline:

        YOLO person tracking
                 ↓
             Person bbox
                 ↓
        InsightFace detection
                 ↓
          ArcFace embedding
                 ↓
        Identity matching
                 ↓
        Track ID + identity
    """

    PERSON_CLASS = "person"

    def __init__(
        self,
        matcher: FaceMatcher,
        detector: Optional[
            FaceDetector
        ] = None,
    ):

        self.detector = (
            detector
            if detector is not None
            else FaceDetector()
        )

        self.matcher = matcher

    def recognize(
        self,
        frame,
        tracks: List[Track],
    ) -> List[FaceTrackResult]:

        faces = self.detector.detect(
            frame
        )

        results = []

        person_tracks = [
            track
            for track in tracks
            if track.class_name
            == self.PERSON_CLASS
        ]

        for face in faces:

            track = self._find_person_track(
                face,
                person_tracks,
            )

            if track is None:
                continue

            match = self.matcher.match(
                face.embedding
            )

            if (
                match is not None
                and match["matched"]
            ):

                identity_id = (
                    match["identity_id"]
                )

                name = match["name"]

                similarity = float(
                    match["similarity"]
                )

                status = "AUTHORIZED"

            else:

                identity_id = None
                name = None

                if match is not None:

                    similarity = float(
                        match["similarity"]
                    )

                else:

                    similarity = 0.0

                status = "UNKNOWN"

            results.append(
                FaceTrackResult(
                    track_id=track.track_id,
                    face_bbox=face.bbox,
                    face_confidence=(
                        face.confidence
                    ),
                    identity_id=identity_id,
                    name=name,
                    similarity=similarity,
                    recognition_status=status,
                )
            )

        return results

    @staticmethod
    def _find_person_track(
        face: FaceDetection,
        tracks: List[Track],
    ) -> Optional[Track]:
        """
        Associate a face with the person track
        whose bounding box contains the face center.
        """

        fx1, fy1, fx2, fy2 = (
            face.bbox
        )

        face_center_x = (
            fx1 + fx2
        ) / 2

        face_center_y = (
            fy1 + fy2
        ) / 2

        best_track = None
        best_area = None

        for track in tracks:

            tx1, ty1, tx2, ty2 = (
                track.bbox
            )

            if not (
                tx1
                <= face_center_x
                <= tx2
            ):
                continue

            if not (
                ty1
                <= face_center_y
                <= ty2
            ):
                continue

            area = (
                (tx2 - tx1)
                * (ty2 - ty1)
            )

            if (
                best_area is None
                or area < best_area
            ):

                best_area = area
                best_track = track

        return best_track