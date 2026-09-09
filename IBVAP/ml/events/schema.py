from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional


@dataclass
class MLEvent:
    """
    Standard event produced by the IBVAP ML layer.

    Every ML module can eventually publish its
    results through this common structure.
    """

    event_type: str

    camera_id: str

    confidence: float

    timestamp: str = field(
        default_factory=lambda: (
            datetime.now(timezone.utc).isoformat()
        )
    )

    severity: str = "INFO"

    track_id: Optional[int] = None

    object_type: Optional[str] = None

    details: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the event into a JSON-compatible
        dictionary.
        """

        return {
            "event_type": self.event_type,
            "camera_id": self.camera_id,
            "timestamp": self.timestamp,
            "confidence": float(
                self.confidence
            ),
            "severity": self.severity,
            "track_id": self.track_id,
            "object_type": self.object_type,
            "details": self.details,
        }