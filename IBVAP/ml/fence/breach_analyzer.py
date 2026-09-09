from dataclasses import dataclass
from typing import Optional

from .config import (
    TEMPORAL_CONFIRMATION_FRAMES,
    BREACH_CONFIDENCE_THRESHOLD,
)


@dataclass
class BreachAnalysis:
    """
    Result of temporal fence-breach analysis.
    """

    confirmed: bool

    confidence: float

    consecutive_frames: int

    reason: Optional[str]


class FenceBreachAnalyzer:
    """
    IBVAP physical fence breach analyzer.

    Converts frame-level abnormal observations
    into a temporally confirmed security event.

    A single abnormal frame does not immediately
    trigger an alarm.
    """

    def __init__(
        self,
        confirmation_frames: int = (
            TEMPORAL_CONFIRMATION_FRAMES
        ),
        confidence_threshold: float = (
            BREACH_CONFIDENCE_THRESHOLD
        ),
    ):

        self.confirmation_frames = (
            confirmation_frames
        )

        self.confidence_threshold = (
            confidence_threshold
        )

        self.consecutive_abnormal_frames = 0

        self.last_confidence = 0.0

        self.confirmed = False

    def update(
        self,
        abnormal: bool,
        confidence: float,
    ) -> BreachAnalysis:
        """
        Update the temporal breach state.

        Parameters
        ----------
        abnormal:
            Whether the current frame contains
            an abnormal fence condition.

        confidence:
            Confidence assigned to the abnormal
            observation.
        """

        confidence = max(
            0.0,
            min(1.0, float(confidence)),
        )

        if (
            abnormal
            and confidence
            >= self.confidence_threshold
        ):

            self.consecutive_abnormal_frames += 1

            self.last_confidence = confidence

        else:

            self.consecutive_abnormal_frames = 0

            self.last_confidence = 0.0

            self.confirmed = False

        if (
            self.consecutive_abnormal_frames
            >= self.confirmation_frames
        ):

            self.confirmed = True

        if self.confirmed:

            return BreachAnalysis(
                confirmed=True,
                confidence=self.last_confidence,
                consecutive_frames=(
                    self.consecutive_abnormal_frames
                ),
                reason=(
                    "Fence abnormality persisted "
                    "across consecutive frames."
                ),
            )

        return BreachAnalysis(
            confirmed=False,
            confidence=self.last_confidence,
            consecutive_frames=(
                self.consecutive_abnormal_frames
            ),
            reason=None,
        )

    def reset(self):
        """
        Reset the temporal state.
        """

        self.consecutive_abnormal_frames = 0

        self.last_confidence = 0.0

        self.confirmed = False