"""Image-key configuration compatible with RLinf's pinned OpenPI package."""

import dataclasses

from openpi.models.pi0_config import Pi0Config
from openpi.shared import array_typing as at


@dataclasses.dataclass(frozen=True)
class ImagePi0Config(Pi0Config):
    """Allow an explicit view layout without changing checkpoint parameters."""

    image_keys: tuple[str, ...] = (
        "base_0_rgb",
        "left_wrist_0_rgb",
        "right_wrist_0_rgb",
    )

    def inputs_spec(self, *, batch_size: int = 1):
        """Extend image and mask specs using the original per-view specs."""
        obs, actions = super().inputs_spec(batch_size=batch_size)
        image = next(iter(obs.images.values()))
        mask = next(iter(obs.image_masks.values()))
        with at.disable_typechecking():
            return obs.replace(
                images=dict.fromkeys(self.image_keys, image),
                image_masks=dict.fromkeys(self.image_keys, mask),
            ), actions
