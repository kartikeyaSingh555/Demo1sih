from typing import List, Optional

from ml.detection.detector import ObjectDetector
from ml.tracking.tracker import ObjectTracker

from ml.camera_health.monitor import CameraHealthMonitor

from ml.anpr.vehicle_plate import VehiclePlateAnalyzer

from ml.face.face_detector import FaceDetector
from ml.face.face_matcher import FaceMatcher
from ml.face.face_tracking import FaceTrackingRecognizer

from ml.weapon.detector import WeaponDetector

from ml.drone.detector import DroneDetector

from ml.fence.fence_detector import FenceDetector

from ml.events.event_builder import MLEventBuilder


class IBVAPVideoProcessor:
    """
    Central IBVAP ML video processing controller.

    Pipeline:

        CCTV / Video Frame
                |
                +--> Camera Health
                |
                +--> Object Detection
                |
                +--> Object Tracking
                |
                +--> Face Recognition
                |
                +--> ANPR
                |
                +--> Weapon Detection
                |
                +--> Drone Detection
                |
                +--> Fence Monitoring
                |
                v
        Unified ML Events

    Lightweight models run every frame.

    Heavy analysis modules can run at a configurable
    frame interval to improve real-time performance.
    """

    def __init__(
        self,
        camera_id: str,
        heavy_interval: int = 5,
    ):
        self.camera_id = camera_id

        self.heavy_interval = max(
            1,
            int(heavy_interval),
        )

        self.frame_count = 0

        print("\n======================================")
        print("       LOADING IBVAP ML ENGINE")
        print("======================================")

        print("\n[1/8] Loading object detector...")
        self.detector = ObjectDetector()

        print("\n[2/8] Loading object tracker...")
        self.tracker = ObjectTracker()

        print("\n[3/8] Loading camera health monitor...")
        self.camera_health = CameraHealthMonitor()

        print("\n[4/8] Loading ANPR analyzer...")
        self.anpr = VehiclePlateAnalyzer()

        print("\n[5/8] Loading face recognition...")

        self.face_detector = FaceDetector()

        self.face_matcher = FaceMatcher(
    threshold=0.45  
)

        self.face_tracker = FaceTrackingRecognizer(
        matcher=self.face_matcher,
        detector=self.face_detector,
    )

        print("\n[6/8] Loading weapon detector...")
        self.weapon_detector = WeaponDetector()

        print("\n[7/8] Loading drone detector...")
        self.drone_detector = DroneDetector()

        print("\n[8/8] Loading fence detector...")
        self.fence_detector = FenceDetector()

        self.event_builder = MLEventBuilder()

        print("\n======================================")
        print("       IBVAP ML ENGINE READY")
        print("======================================")

    def _process_camera_health(
        self,
        frame,
        events: List,
    ):
        health = self.camera_health.analyze(
            frame
        )

        if health.status == "HEALTHY":
            return health

        severity = (
            "CRITICAL"
            if health.status == "OFFLINE"
            else "MEDIUM"
        )

        event = (
            self.event_builder
            .from_camera_health(
                camera_id=self.camera_id,
                status=health.status,
                brightness=health.brightness,
                sharpness=health.sharpness,
                black_ratio=health.black_ratio,
                issues=health.issues,
                severity=severity,
            )
        )

        events.append(event)

        return health

    def _process_detection(
        self,
        frame,
        events: List,
    ):
        detections = self.detector.detect(
            frame
        )

        for detection in detections:

            event = self.event_builder.build(
                event_type="OBJECT_DETECTED",
                camera_id=self.camera_id,
                confidence=detection.confidence,
                severity="INFO",
                object_type=detection.class_name,
                details={
                    "class_id": detection.class_id,
                    "bbox": detection.bbox,
                },
            )

            events.append(event)

        return detections

    def _process_tracking(
        self,
        frame,
        events: List,
    ):
        tracks = self.tracker.track(
            frame
        )

        for track in tracks:

            event = self.event_builder.build(
                event_type="OBJECT_TRACKED",
                camera_id=self.camera_id,
                confidence=track.confidence,
                severity="INFO",
                track_id=track.track_id,
                object_type=track.class_name,
                details={
                    "class_id": track.class_id,
                    "bbox": track.bbox,
                },
            )

            events.append(event)

        return tracks

    def _process_face(
        self,
        frame,
        tracks,
        events: List,
    ):
        results = self.face_tracker.recognize(
            frame,
            tracks,
        )

        for result in results:

            event = self.event_builder.from_face(
                camera_id=self.camera_id,
                track_id=result.track_id,
                identity_id=result.identity_id,
                name=result.name,
                status=result.recognition_status,
                face_confidence=result.face_confidence,
                match_confidence=result.similarity,
                bbox=result.face_bbox,
            )

            events.append(event)

    def _process_anpr(
        self,
        frame,
        tracks,
        events: List,
    ):
        results = self.anpr.analyze(
            frame,
            tracks,
        )

        for result in results:

            event = self.event_builder.build(
                event_type="ANPR_DETECTED",
                camera_id=self.camera_id,
                confidence=(
                    result.ocr_confidence
                    if result.ocr_confidence
                    is not None
                    else result.plate_confidence
                ),
                severity="INFO",
                track_id=result.track_id,
                object_type=result.vehicle_class,
                details={
                    "plate_text": result.plate_text,
                    "ocr_confidence": (
                        result.ocr_confidence
                    ),
                    "plate_confidence": (
                        result.plate_confidence
                    ),
                    "vehicle_bbox": (
                        result.vehicle_bbox
                    ),
                    "plate_bbox": (
                        result.plate_bbox
                    ),
                },
            )

            events.append(event)

    def _process_weapon(
        self,
        frame,
        events: List,
    ):
        detections = self.weapon_detector.detect(
            frame
        )

        for detection in detections:

            event = self.event_builder.from_weapon(
                camera_id=self.camera_id,
                weapon_type=detection.class_name,
                confidence=detection.confidence,
                bbox=detection.bbox,
            )

            events.append(event)

    def _process_drone(
        self,
        frame,
        events: List,
    ):
        detections = self.drone_detector.detect(
            frame
        )

        for detection in detections:

            event = self.event_builder.from_drone(
                camera_id=self.camera_id,
                confidence=detection.confidence,
                bbox=detection.bbox,
            )

            events.append(event)

    def _process_fence(
        self,
        frame,
        events: List,
    ):
        """
        Runs fence condition analysis.

        The fence detector must have a reference
        frame before analysis can begin.

        Reference initialization is handled by
        initialize_fence_reference().
        """

        if self.fence_detector.reference is None:
            return

        condition = self.fence_detector.detect(
            frame
        )

        if not condition.abnormal:
            return

        confidence = min(
            condition.change_score / 0.50,
            1.0,
        )

        event = self.event_builder.from_fence(
            camera_id=self.camera_id,
            breach_type="DAMAGED",
            confidence=confidence,
            severity="HIGH",
            bbox=condition.bbox,
        )

        events.append(event)

    def initialize_fence_reference(
        self,
        frame,
    ):
        """
        Establish the normal physical fence
        reference frame.
        """

        self.fence_detector.set_reference(
            frame
        )

    def process_frame(
        self,
        frame,
    ) -> List:

        events = []

        if frame is None:
            return events

        self.frame_count += 1

        # --------------------------------------
        # EVERY FRAME
        # --------------------------------------

        self._process_camera_health(
            frame,
            events,
        )

        self._process_detection(
            frame,
            events,
        )

        tracks = self._process_tracking(
            frame,
            events,
        )

        # --------------------------------------
        # HEAVY ANALYSIS
        # --------------------------------------

        if (
            self.frame_count
            % self.heavy_interval
            == 0
        ):

            self._process_face(
                frame,
                tracks,
                events,
            )

            self._process_anpr(
                frame,
                tracks,
                events,
            )

            self._process_weapon(
                frame,
                events,
            )

            self._process_drone(
                frame,
                events,
            )

            self._process_fence(
                frame,
                events,
            )

        return events

