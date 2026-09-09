from dataclasses import dataclass
from typing import Optional, Dict, Any

from ml.face.face_tracking import FaceTrackResult


@dataclass
class FaceEvent:
    """
    Standard IBVAP face recognition event.

    This event is produced by the ML layer and can
    later be consumed by the backend and AI layers.
    """

    event_type: str

    track_id: int

    identity_id: Optional[str]

    name: Optional[str]

    recognition_status: str

    face_confidence: float

    match_confidence: float

    face_bbox: tuple

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the event into a JSON-compatible
        dictionary.
        """

        return {
            "event_type": self.event_type,
            "track_id": self.track_id,
            "identity_id": self.identity_id,
            "name": self.name,
            "recognition_status": self.recognition_status,
            "face_confidence": self.face_confidence,
            "match_confidence": self.match_confidence,
            "face_bbox": self.face_bbox,
        }


class FaceEventGenerator:
    """
    Converts FaceTrackResult objects into standardized
    IBVAP FaceEvent objects.
    """

    EVENT_TYPE = "FACE_RECOGNITION"

    def generate(
        self,
        result: FaceTrackResult,
    ) -> FaceEvent:
        """
        Generate one face recognition event.
        """

        return FaceEvent(
            event_type=self.EVENT_TYPE,

            track_id=result.track_id,

            identity_id=result.identity_id,

            name=result.name,

            recognition_status=(
                result.recognition_status
            ),

            face_confidence=(
                float(result.face_confidence)
            ),

            match_confidence=(
                float(result.similarity)
            ),

            face_bbox=result.face_bbox,
        )

    def generate_many(
        self,
        results,
    ):
        """
        Generate events for multiple recognized
        faces.
        """

        return [
            self.generate(result)
            for result in results
        ]