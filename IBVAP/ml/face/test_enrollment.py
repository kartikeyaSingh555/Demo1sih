from ml.face.enrollment import (
    FaceEnrollmentManager,
)


def main():

    image_path = (
        "data/demo/face_test.jpg"
    )

    identity_id = "PERSON_001"

    name = "Demo Authorized Person"

    print("\n======================================")
    print("       IBVAP FACE ENROLLMENT TEST")
    print("======================================")

    print(
        "\n[1/3] Starting enrollment manager..."
    )

    manager = FaceEnrollmentManager()

    print(
        "\n[2/3] Enrolling authorized person..."
    )

    result = manager.enroll(
        identity_id=identity_id,
        name=name,
        image_path=image_path,
    )

    print(
        "\n========== ENROLLMENT RESULT =========="
    )

    print(
        f"Success           : "
        f"{result['success']}"
    )

    if not result["success"]:

        print(
            f"Message           : "
            f"{result['message']}"
        )

        return

    print(
        f"Identity ID       : "
        f"{result['identity_id']}"
    )

    print(
        f"Name              : "
        f"{result['name']}"
    )

    print(
        f"Embedding dimension: "
        f"{result['embedding_dimension']}"
    )

    print(
        f"Database           : "
        f"{result['database_path']}"
    )

    print(
        "\n[3/3] Checking stored identity..."
    )

    identities = (
        manager.list_identities()
    )

    print(
        "\n========== STORED IDENTITIES =========="
    )

    for identity in identities:

        print(
            f"ID: {identity['identity_id']} "
            f"| Name: {identity['name']}"
        )

    print(
        "\n========================================"
    )


if __name__ == "__main__":
    main()