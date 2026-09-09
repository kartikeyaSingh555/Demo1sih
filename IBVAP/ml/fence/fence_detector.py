from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from .config import (
    FENCE_CHANGE_THRESHOLD,
)


@dataclass
class FenceCondition:
    """
    Represents the current physical condition
    of the monitored fence region.
    """

    abnormal: bool

    change_score: float

    bbox: Optional[Tuple[int, int, int, int]]


class FenceDetector:
    """
    IBVAP physical fence condition detector.

    Maintains a reference image of the normal
    fence and compares subsequent frames against it.

    This component detects significant visual
    changes.

    It does not independently decide whether the
    change is a confirmed security threat.
    """

    def __init__(
        self,
        change_threshold: float = FENCE_CHANGE_THRESHOLD,
    ):

        self.change_threshold = (
            change_threshold
        )

        self.reference = None

    def set_reference(
        self,
        frame,
    ):
        """
        Store a normal/reference fence frame.
        """

        if frame is None:
            raise ValueError(
                "Reference frame cannot be None."
            )

        self.reference = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

    def detect(
        self,
        frame,
    ) -> FenceCondition:
        """
        Compare the current frame with the
        normal fence reference.
        """

        if frame is None:
            return FenceCondition(
                abnormal=False,
                change_score=0.0,
                bbox=None,
            )

        if self.reference is None:
            raise RuntimeError(
                "Fence reference has not been set."
            )

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

        if gray.shape != self.reference.shape:
            gray = cv2.resize(
                gray,
                (
                    self.reference.shape[1],
                    self.reference.shape[0],
                ),
            )

        difference = cv2.absdiff(
            self.reference,
            gray,
        )

        _, thresholded = cv2.threshold(
            difference,
            30,
            255,
            cv2.THRESH_BINARY,
        )

        change_score = float(
            np.count_nonzero(
                thresholded
            )
            / thresholded.size
        )

        abnormal = (
            change_score
            >= self.change_threshold
        )

        bbox = None

        if abnormal:

            contours, _ = cv2.findContours(
                thresholded,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            if contours:

                largest = max(
                    contours,
                    key=cv2.contourArea,
                )

                x, y, w, h = (
                    cv2.boundingRect(
                        largest
                    )
                )

                bbox = (
                    int(x),
                    int(y),
                    int(x + w),
                    int(y + h),
                )

        return FenceCondition(
            abnormal=abnormal,
            change_score=change_score,
            bbox=bbox,
        )