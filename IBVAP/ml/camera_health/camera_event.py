from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class CameraHealthEvent:
    """
    Standard IBVAP camera-health event.
    """

    event_type: str

    camera_id: str

    status: str

    brightness: float

    sharpness: float

    black_ratio: float

    issues: List[str]

    severity: str

    message: str

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the event to a JSON-compatible dictionary.
        """

        return {
            "event_type": self.event_type,
            "camera_id": self.camera_id,
            "status": self.status,
            "brightness": self.brightness,
            "sharpness": self.sharpness,
            "black_ratio": self.black_ratio,
            "issues": self.issues,
            "severity": self.severity,
            "message": self.message,
        }


class CameraHealthEventGenerator:
    """
    Generates standardized IBVAP camera health events.
    """

    EVENT_TYPE = "CAMERA_HEALTH"

    def generate(
        self,
        health,
        camera_id: str,
    ) -> Optional[CameraHealthEvent]:
        """
        Generate an event.

        Healthy cameras do not generate an alert event.
        """

        if health.status == "HEALTHY":

            return None

        if health.status == "OFFLINE":

            severity = "CRITICAL"

        elif health.status == "DEGRADED":

            severity = "HIGH"

        else:

            severity = "MEDIUM"

        if health.issues:

            message = (
                "Camera health issue detected: "
                + ", ".join(
                    health.issues
                )
            )

        else:

            message = (
                "Camera health degraded."
            )

        return CameraHealthEvent(
            event_type=self.EVENT_TYPE,
            camera_id=camera_id,
            status=health.status,
            brightness=float(
                health.brightness
            ),
            sharpness=float(
                health.sharpness
            ),
            black_ratio=float(
                health.black_ratio
            ),
            issues=health.issues,
            severity=severity,
            message=message,
        )