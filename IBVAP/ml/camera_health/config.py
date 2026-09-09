"""
IBVAP Camera Health Monitoring Configuration
"""


# Minimum average brightness.
# Below this, the frame may be considered too dark.

MIN_BRIGHTNESS = 25.0


# Maximum average brightness.
# Above this, the frame may be considered overexposed.

MAX_BRIGHTNESS = 235.0


# Blur threshold.
#
# Lower Laplacian variance means the image
# contains less detail and may be heavily blurred.

MIN_SHARPNESS = 30.0


# Percentage of pixels that must be extremely
# dark before considering a frame potentially black.

BLACK_FRAME_RATIO = 0.90


# Health states.

HEALTHY = "HEALTHY"

WARNING = "WARNING"

DEGRADED = "DEGRADED"

OFFLINE = "OFFLINE"