"""
IBVAP Physical Fence Monitoring Configuration
"""


# Minimum number of consecutive frames in which
# an abnormal condition must remain before it
# can become a confirmed event.

TEMPORAL_CONFIRMATION_FRAMES = 5


# Minimum confidence required before an event
# is reported.

BREACH_CONFIDENCE_THRESHOLD = 0.60


# Fence visual-change threshold.

FENCE_CHANGE_THRESHOLD = 0.20


# Number of frames used to establish the
# normal/reference fence appearance.

REFERENCE_FRAMES = 30


# Event types.

FENCE_BREACH_EVENT = "FENCE_BREACH"

FENCE_DAMAGE_EVENT = "FENCE_DAMAGE"