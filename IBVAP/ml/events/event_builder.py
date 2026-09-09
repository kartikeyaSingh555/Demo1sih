from typing import Any, Dict, Optional

from .schema import MLEvent


class MLEventBuilder:
    """
    Creates standardized IBVAP ML events.

    This acts as the bridge between individual
    ML modules and the AI/backend layers.
    """

    def build(
        self,
        event_type: str,
        camera_id: str,
        confidence: float = 1.0,
        severity: str = "INFO",
        track_id: Optional[int] = None,
        object_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> MLEvent:

        if details is None:
            details = {}

        return MLEvent(
            event_type=event_type,
            camera_id=camera_id,
            confidence=float(confidence),
            severity=severity.upper(),
            track_id=track_id,
            object_type=object_type,
            details=details,
        )

    def from_weapon(
        self,
        camera_id: str,
        weapon_type: str,
        confidence: float,
        bbox=None,
    ) -> MLEvent:

        return self.build(
            event_type="WEAPON_DETECTED",
            camera_id=camera_id,
            confidence=confidence,
            severity="HIGH",
            object_type=weapon_type,
            details={
                "weapon_type": weapon_type,
                "bbox": bbox,
            },
        )

    def from_drone(
        self,
        camera_id: str,
        confidence: float,
        bbox=None,
    ) -> MLEvent:

        return self.build(
            event_type="DRONE_DETECTED",
            camera_id=camera_id,
            confidence=confidence,
            severity="HIGH",
            object_type="drone",
            details={
                "bbox": bbox,
            },
        )

    def from_face(
        self,
        camera_id: str,
        track_id: int,
        identity_id: Optional[str],
        name: Optional[str],
        status: str,
        face_confidence: float,
        match_confidence: float,
        bbox=None,
    ) -> MLEvent:

        status = status.upper()

        if status == "UNKNOWN":
            severity = "MEDIUM"
        else:
            severity = "INFO"

        return self.build(
            event_type="FACE_RECOGNITION",
            camera_id=camera_id,
            confidence=match_confidence,
            severity=severity,
            track_id=track_id,
            object_type="person",
            details={
                "identity_id": identity_id,
                "name": name,
                "recognition_status": status,
                "face_confidence": face_confidence,
                "match_confidence": match_confidence,
                "bbox": bbox,
            },
        )

    def from_fence(
        self,
        camera_id: str,
        breach_type: str,
        confidence: float,
        severity: str = "CRITICAL",
        bbox=None,
        track_id=None,
    ) -> MLEvent:

        breach_type = breach_type.upper()

        if breach_type in {
            "CUT",
            "BROKEN",
            "DAMAGED",
            "MISSING_SECTION",
        }:
            event_type = "FENCE_DAMAGE"
        else:
            event_type = "FENCE_BREACH"

        return self.build(
            event_type=event_type,
            camera_id=camera_id,
            confidence=confidence,
            severity=severity,
            track_id=track_id,
            object_type="fence",
            details={
                "breach_type": breach_type,
                "bbox": bbox,
            },
        )

    def from_camera_health(
        self,
        camera_id: str,
        status: str,
        brightness: float,
        sharpness: float,
        black_ratio: float,
        issues,
        severity: str,
    ) -> MLEvent:

        return self.build(
            event_type="CAMERA_HEALTH",
            camera_id=camera_id,
            confidence=1.0,
            severity=severity,
            object_type="camera",
            details={
                "status": status,
                "brightness": brightness,
                "sharpness": sharpness,
                "black_ratio": black_ratio,
                "issues": issues,
            },
        )