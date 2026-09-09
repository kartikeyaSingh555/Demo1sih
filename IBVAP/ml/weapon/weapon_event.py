from dataclasses import dataclass
from typing import Dict, Any, Tuple

from ml.weapon.detector import WeaponDetection


@dataclass
class WeaponEvent:
    """
    Standard IBVAP weapon detection event.

    This event describes a detected weapon-like
    object. It does not determine threat level.
    """

    event_type: str

    weapon_type: str

    confidence: float

    bbox: Tuple[int, int, int, int]

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the event into a JSON-compatible
        dictionary.
        """

        return {
            "event_type": self.event_type,
            "weapon_type": self.weapon_type,
            "confidence": self.confidence,
            "bbox": self.bbox,
        }


class WeaponEventGenerator:
    """
    Converts WeaponDetection objects into
    standardized IBVAP WeaponEvent objects.
    """

    EVENT_TYPE = "WEAPON_DETECTED"

    def generate(
        self,
        detection: WeaponDetection,
    ) -> WeaponEvent:
        """
        Generate one weapon detection event.
        """

        return WeaponEvent(
            event_type=self.EVENT_TYPE,

            weapon_type=detection.class_name,

            confidence=float(
                detection.confidence
            ),

            bbox=detection.bbox,
        )

    def generate_many(
        self,
        detections,
    ):
        """
        Generate events for multiple weapon
        detections.
        """

        return [
            self.generate(detection)
            for detection in detections
        ]