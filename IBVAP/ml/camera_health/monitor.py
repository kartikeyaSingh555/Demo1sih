from dataclasses import dataclass
from typing import List

import cv2
import numpy as np

from .config import (
    MIN_BRIGHTNESS,
    MAX_BRIGHTNESS,
    MIN_SHARPNESS,
    BLACK_FRAME_RATIO,
    HEALTHY,
    WARNING,
    DEGRADED,
    OFFLINE,
)


@dataclass
class CameraHealth:
    """
    Represents the health condition of a CCTV camera.
    """

    status: str

    brightness: float

    sharpness: float

    black_ratio: float

    issues: List[str]


class CameraHealthMonitor:
    """
    IBVAP CCTV camera health monitoring module.

    Checks:
        - Missing frames
        - Black screen
        - Excessive darkness
        - Excessive brightness
        - Image blur

    This module does not detect security threats.
    It determines whether the camera feed itself
    is usable.
    """

    def __init__(
        self,
        min_brightness: float = MIN_BRIGHTNESS,
        max_brightness: float = MAX_BRIGHTNESS,
        min_sharpness: float = MIN_SHARPNESS,
        black_frame_ratio: float = BLACK_FRAME_RATIO,
    ):

        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_sharpness = min_sharpness
        self.black_frame_ratio = black_frame_ratio

    def analyze(self, frame) -> CameraHealth:
        """
        Analyze one CCTV frame.
        """

        # ------------------------------------------
        # Frame missing
        # ------------------------------------------

        if frame is None:

            return CameraHealth(
                status=OFFLINE,
                brightness=0.0,
                sharpness=0.0,
                black_ratio=1.0,
                issues=[
                    "NO_FRAME"
                ],
            )

        # ------------------------------------------
        # Invalid frame
        # ------------------------------------------

        if not isinstance(
            frame,
            np.ndarray,
        ):

            return CameraHealth(
                status=OFFLINE,
                brightness=0.0,
                sharpness=0.0,
                black_ratio=1.0,
                issues=[
                    "INVALID_FRAME"
                ],
            )

        if frame.size == 0:

            return CameraHealth(
                status=OFFLINE,
                brightness=0.0,
                sharpness=0.0,
                black_ratio=1.0,
                issues=[
                    "EMPTY_FRAME"
                ],
            )

        # ------------------------------------------
        # Convert to grayscale
        # ------------------------------------------

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

        # ------------------------------------------
        # Brightness
        # ------------------------------------------

        brightness = float(
            np.mean(gray)
        )

        # ------------------------------------------
        # Sharpness
        # ------------------------------------------

        sharpness = float(
            cv2.Laplacian(
                gray,
                cv2.CV_64F,
            ).var()
        )

        # ------------------------------------------
        # Black frame detection
        # ------------------------------------------

        black_pixels = np.sum(
            gray <= 10
        )

        total_pixels = gray.size

        black_ratio = float(
            black_pixels / total_pixels
        )

        issues = []

        # ------------------------------------------
        # Check black frame
        # ------------------------------------------

        if (
            black_ratio
            >= self.black_frame_ratio
        ):

            issues.append(
                "BLACK_SCREEN"
            )

        # ------------------------------------------
        # Check brightness
        # ------------------------------------------

        if brightness < self.min_brightness:

            issues.append(
                "TOO_DARK"
            )

        elif brightness > self.max_brightness:

            issues.append(
                "OVEREXPOSED"
            )

        # ------------------------------------------
        # Check blur
        # ------------------------------------------

        if sharpness < self.min_sharpness:

            issues.append(
                "BLURRED"
            )

        # ------------------------------------------
        # Determine overall status
        # ------------------------------------------

        if (
            "BLACK_SCREEN"
            in issues
        ):

            status = OFFLINE

        elif len(issues) >= 2:

            status = DEGRADED

        elif len(issues) == 1:

            status = WARNING

        else:

            status = HEALTHY

        return CameraHealth(
            status=status,
            brightness=brightness,
            sharpness=sharpness,
            black_ratio=black_ratio,
            issues=issues,
        )