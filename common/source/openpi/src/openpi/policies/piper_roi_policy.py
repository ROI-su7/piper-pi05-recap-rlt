"""Right-wrist ROI view for the Piper connect-strings task.

The right gripper holds the splitter socket, so the socket mouth sits at a fixed place in the right wrist camera
(640x480). Cropping that region before the usual 224x224 resize roughly doubles the pixels on the mating junction
(about 1.2 px/mm -> 2.3 px/mm). The crop is fed as a fourth image next to the three original cameras.

Training data (LeRobot, raw 640x480 frames) is cropped here. At inference the Piper client already resizes every
camera to 224x224, so it must send the ROI itself as ``cam_right_wrist_roi`` (cropped from the raw frame with
``RIGHT_WRIST_ROI_BOX`` and then resized with ``resize_with_pad``); see ``examples/piper_real/env.py``.
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model
from openpi.policies import aloha_policy

# (x0, y0, x1, y1) in the raw right wrist image; 320x240 keeps the 4:3 aspect of the camera.
RIGHT_WRIST_ROI_BOX = (165, 130, 485, 370)
RAW_IMAGE_HW = (480, 640)
# Input camera name sent by the client, and the model image key. The key contains "wrist", so training applies
# wrist-style augmentation (color jitter only, no crop/rotation that would move the junction).
ROI_CAMERA = "cam_right_wrist_roi"
ROI_IMAGE_KEY = "right_wrist_roi_0_rgb"
ROI_IMAGE_KEYS = (*_model.IMAGE_KEYS, ROI_IMAGE_KEY)


def _to_hwc_uint8(img) -> np.ndarray:
    """Same conversion as ``aloha_policy._decode_aloha``: float [0, 1] -> uint8, CHW -> HWC."""
    img = np.asarray(img)
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    return einops.rearrange(img, "c h w -> h w c")


def crop_right_wrist_roi(image_chw, box: tuple[int, int, int, int] = RIGHT_WRIST_ROI_BOX) -> np.ndarray:
    """Crop the ROI from a raw ``[3, 480, 640]`` right wrist image, keeping CHW layout and dtype."""
    image_chw = np.asarray(image_chw)
    if image_chw.ndim != 3 or tuple(image_chw.shape[1:]) != RAW_IMAGE_HW:
        raise ValueError(
            f"The right wrist ROI needs the raw {RAW_IMAGE_HW[1]}x{RAW_IMAGE_HW[0]} image in CHW layout, got shape "
            f"{image_chw.shape}. A client that resizes cameras before sending must send '{ROI_CAMERA}' itself "
            "(examples/piper_real: --right-wrist-roi)."
        )
    x0, y0, x1, y1 = box
    return image_chw[:, y0:y1, x0:x1]


@dataclasses.dataclass(frozen=True)
class PiperRoiAlohaInputs(transforms.DataTransformFn):
    """``AlohaInputs`` plus the right wrist ROI as a fourth image.

    Uses ``images[ROI_CAMERA]`` when the client sends it, otherwise crops it from the raw ``cam_right_wrist``.
    """

    adapt_to_pi: bool = True
    box: tuple[int, int, int, int] = RIGHT_WRIST_ROI_BOX

    def __call__(self, data: dict) -> dict:
        images = dict(data["images"])
        roi = images.pop(ROI_CAMERA, None)
        if roi is None:
            if "cam_right_wrist" not in images:
                raise ValueError(f"Need '{ROI_CAMERA}' or a raw 'cam_right_wrist' image, got {tuple(images)}")
            roi = crop_right_wrist_roi(images["cam_right_wrist"], self.box)
        inputs = aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)({**data, "images": images})
        inputs["image"][ROI_IMAGE_KEY] = _to_hwc_uint8(roi)
        inputs["image_mask"][ROI_IMAGE_KEY] = np.True_
        return inputs
