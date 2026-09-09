import json
from pathlib import Path
from typing import Optional

import numpy as np

from ml.face.face_detector import FaceDetector


class FaceEnrollmentManager:
    """
    IBVAP face enrollment manager.

    Converts an authorized person's face image
    into an ArcFace embedding and stores it locally.

    This is a prototype storage layer.

    Later, the backend/database layer can replace
    this file-based storage with MongoDB.
    """

    def __init__(
        self,
        database_path: str = (
            "data/watchlist/embeddings/"
            "face_database.json"
        ),
    ):

        self.database_path = Path(
            database_path
        )

        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.detector = FaceDetector()

    def _load_database(self):
        """
        Load the existing face database.
        """

        if not self.database_path.exists():
            return {}

        with open(
            self.database_path,
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(file)

    def _save_database(
        self,
        database,
    ):
        """
        Save the face database.
        """

        with open(
            self.database_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                database,
                file,
                indent=4,
            )

    def enroll(
        self,
        identity_id: str,
        name: str,
        image_path: str,
    ) -> dict:
        """
        Enroll one authorized identity.

        The image must contain a detectable face.
        """

        image = self._load_image(
            image_path
        )

        if image is None:

            return {
                "success": False,
                "message": (
                    f"Could not load image: "
                    f"{image_path}"
                ),
            }

        detections = self.detector.detect(
            image
        )

        if not detections:

            return {
                "success": False,
                "message": (
                    "No face detected."
                ),
            }

        if len(detections) > 1:

            return {
                "success": False,
                "message": (
                    "Multiple faces detected. "
                    "Enrollment requires exactly "
                    "one face."
                ),
            }

        face = detections[0]

        embedding = face.embedding.astype(
            np.float32
        )

        database = self._load_database()

        database[identity_id] = {
            "identity_id": identity_id,
            "name": name,
            "embedding": embedding.tolist(),
            "embedding_dimension": len(
                embedding
            ),
            "source_image": image_path,
        }

        self._save_database(
            database
        )

        return {
            "success": True,
            "identity_id": identity_id,
            "name": name,
            "embedding_dimension": len(
                embedding
            ),
            "database_path": str(
                self.database_path
            ),
        }

    def get_identity(
        self,
        identity_id: str,
    ) -> Optional[dict]:
        """
        Retrieve one enrolled identity.
        """

        database = self._load_database()

        return database.get(
            identity_id
        )

    def list_identities(self):
        """
        Return all enrolled identities.
        """

        database = self._load_database()

        identities = []

        for identity_id, record in (
            database.items()
        ):

            identities.append(
                {
                    "identity_id": identity_id,
                    "name": record["name"],
                }
            )

        return identities

    @staticmethod
    def _load_image(
        image_path: str,
    ):

        import cv2

        image = cv2.imread(
            image_path
        )

        return image