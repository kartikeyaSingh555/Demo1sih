import cv2


class VideoSource:
    """
    IBVAP video input source.

    Supports:
        - Local video files
        - CCTV/RTSP streams
        - Camera devices
    """

    def __init__(
        self,
        source,
    ):
        self.source = source
        self.capture = None

    def open(self):
        """
        Open the configured video source.
        """

        self.capture = cv2.VideoCapture(
            self.source
        )

        if not self.capture.isOpened():
            raise RuntimeError(
                f"Could not open video source: "
                f"{self.source}"
            )

    def read(self):
        """
        Read one video frame.

        Returns:
            (success, frame)
        """

        if self.capture is None:
            raise RuntimeError(
                "Video source is not open."
            )

        return self.capture.read()

    def get_fps(self):
        """
        Return source FPS.
        """

        if self.capture is None:
            return 0.0

        fps = self.capture.get(
            cv2.CAP_PROP_FPS
        )

        if fps <= 0:
            return 25.0

        return float(fps)

    def get_width(self):
        """
        Return video width.
        """

        if self.capture is None:
            return 0

        return int(
            self.capture.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

    def get_height(self):
        """
        Return video height.
        """

        if self.capture is None:
            return 0

        return int(
            self.capture.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

    def release(self):
        """
        Release the video source.
        """

        if self.capture is not None:
            self.capture.release()
            self.capture = None