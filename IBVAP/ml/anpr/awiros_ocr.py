import sys
from pathlib import Path

import cv2
import numpy as np
import paddle
from safetensors.numpy import load_file


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PADDLEOCR_PATH = PROJECT_ROOT / "ml" / "anpr" / "paddleocr_core"

MODEL_DIR = PROJECT_ROOT / "models" / "anpr" / "awiros"

MODEL_PATH = MODEL_DIR / "model.safetensors"

DICT_PATH = MODEL_DIR / "en_dict.txt"


# Make PaddleOCR source importable without installing
# the whole PaddleOCR package globally.
if str(PADDLEOCR_PATH) not in sys.path:
    sys.path.insert(0, str(PADDLEOCR_PATH))


from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process


# ---------------------------------------------------------
# Official Awiros model configuration
# ---------------------------------------------------------

CTC_NUM_CLASSES = 64
NRTR_NUM_CLASSES = 67

MODEL_CONFIG = {
    "Architecture": {
        "model_type": "rec",
        "algorithm": "SVTR_HGNet",

        "Transform": None,

        "Backbone": {
            "name": "PPHGNetV2_B4",
            "text_rec": True,
        },

        "Head": {
            "name": "MultiHead",

            "out_channels_list": {
                "CTCLabelDecode": CTC_NUM_CLASSES,
                "NRTRLabelDecode": NRTR_NUM_CLASSES,
            },

            "head_list": [
                {
                    "CTCHead": {
                        "Neck": {
                            "name": "svtr",
                            "dims": 120,
                            "depth": 2,
                            "hidden_dims": 120,
                            "kernel_size": [1, 3],
                            "use_guide": True,
                        },

                        "Head": {
                            "fc_decay": 1e-05,
                        },
                    }
                },

                {
                    "NRTRHead": {
                        "nrtr_dim": 384,
                        "max_text_length": 25,
                    }
                },
            ],
        },
    },
}


IMAGE_SHAPE = [3, 48, 320]


# ---------------------------------------------------------
# Awiros OCR
# ---------------------------------------------------------

class AwirosLicensePlateOCR:
    """
    IBVAP Indian license plate OCR.

    Uses the Awiros ANPR OCR model based on PP-OCRv5 /
    SVTR-HGNet architecture.

    Input:
        BGR OpenCV plate image

    Output:
        OCRResult-like dictionary containing:
            text
            confidence
    """

    def __init__(self, device="cpu"):

        print("Loading IBVAP Awiros ANPR OCR model...")

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Awiros model not found:\n{MODEL_PATH}"
            )

        if not DICT_PATH.exists():
            raise FileNotFoundError(
                f"Awiros dictionary not found:\n{DICT_PATH}"
            )

        # -------------------------------------------------
        # Device
        # -------------------------------------------------

        available_device = "cpu"

        if device == "gpu":
            try:
                if paddle.is_compiled_with_cuda():
                    available_device = "gpu"
                else:
                    print(
                        "GPU requested but CUDA is unavailable."
                        " Falling back to CPU."
                    )
            except Exception:
                print(
                    "Could not initialize GPU."
                    " Falling back to CPU."
                )

        paddle.set_device(available_device)

        self.device = available_device

        print(f"Paddle device: {self.device}")

        # -------------------------------------------------
        # Dictionary
        # -------------------------------------------------

        self.character_dict_path = str(DICT_PATH)

        # -------------------------------------------------
        # Post process
        # -------------------------------------------------

        postprocess_config = {
            "name": "CTCLabelDecode",
            "character_dict_path": self.character_dict_path,
            "use_space_char": True,
        }

        self.post_process = build_post_process(
            postprocess_config
        )

        # -------------------------------------------------
        # Model
        # -------------------------------------------------

        model_config = MODEL_CONFIG["Architecture"]

        self.model = build_model(
            model_config
        )

        # -------------------------------------------------
        # Load SafeTensors
        # -------------------------------------------------

        print("Loading Awiros SafeTensors weights...")

        weights = load_file(
            str(MODEL_PATH)
        )

        loaded_count = 0
        skipped_count = 0

        state_dict = self.model.state_dict()

        for name, value in weights.items():

            if name not in state_dict:
                skipped_count += 1
                continue

            tensor = paddle.to_tensor(
                value
            )

            state_dict[name].set_value(
                tensor
            )

            loaded_count += 1

        self.model.set_state_dict(
            state_dict
        )

        self.model.eval()

        print(
            f"Loaded model tensors: {loaded_count}"
        )

        if skipped_count:
            print(
                f"Skipped model tensors: {skipped_count}"
            )

        print("Awiros ANPR OCR model ready.")

    # -----------------------------------------------------
    # Image preprocessing
    # -----------------------------------------------------

    def preprocess(self, image):

        if image is None:
            return None

        if image.size == 0:
            return None

        img = image.copy()

        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB,
        )

        target_height = IMAGE_SHAPE[1]
        target_width = IMAGE_SHAPE[2]

        height, width = img.shape[:2]

        if height <= 0 or width <= 0:
            return None

        ratio = target_height / float(height)

        resized_width = int(
            width * ratio
        )

        resized_width = min(
            resized_width,
            target_width,
        )

        img = cv2.resize(
            img,
            (
                resized_width,
                target_height,
            ),
            interpolation=cv2.INTER_LINEAR,
        )

        # -------------------------------------------------
        # Right-side black padding
        # -------------------------------------------------

        padded = np.zeros(
            (
                target_height,
                target_width,
                3,
            ),
            dtype=np.uint8,
        )

        padded[
            :,
            :resized_width,
            :
        ] = img

        # -------------------------------------------------
        # Normalize
        # -------------------------------------------------

        padded = padded.astype(
            np.float32
        )

        padded = padded / 255.0

        padded = (
            padded - 0.5
        ) / 0.5

        # HWC -> CHW
        padded = padded.transpose(
            2,
            0,
            1,
        )

        padded = np.expand_dims(
            padded,
            axis=0,
        )

        return paddle.to_tensor(
            padded
        )

    # -----------------------------------------------------
    # OCR
    # -----------------------------------------------------

    def read(self, plate_image):

        image_tensor = self.preprocess(
            plate_image
        )

        if image_tensor is None:
            return None

        with paddle.no_grad():

            outputs = self.model(
                image_tensor
            )

        # -------------------------------------------------
        # CTC output
        # -------------------------------------------------

        if isinstance(outputs, dict):

            if "ctc" in outputs:
                predictions = outputs["ctc"]

            elif "head_out" in outputs:
                predictions = outputs["head_out"]

            else:
                predictions = outputs

        else:
            predictions = outputs

        # -------------------------------------------------
        # Normalize prediction format
        # -------------------------------------------------

        if isinstance(
            predictions,
            (list, tuple)
        ):

            predictions = predictions[0]

        # -------------------------------------------------
        # Decode
        # -------------------------------------------------

        decoded = self.post_process(
            predictions
        )

        if not decoded:
            return None

        first = decoded[0]

        if isinstance(first, (list, tuple)):

            text = str(first[0])
            confidence = float(first[1])

        elif isinstance(first, dict):

            text = str(
                first.get(
                    "text",
                    ""
                )
            )

            confidence = float(
                first.get(
                    "score",
                    0.0
                )
            )

        else:

            text = str(first)
            confidence = 0.0

        text = text.upper()

        # Keep only plate characters.
        text = "".join(
            character
            for character in text
            if character.isalnum()
        )

        if not text:
            return None

        return {
            "text": text,
            "confidence": confidence,
        }