f"""

define other transformations here, these were taken from
the colab notebook by `monai` (link: https://colab.research.google.com/drive/1IqdpUPM_CoKYj6EHNb-IYaCiHvEiM08D#scrollTo=1zUgH23MFiMX)

""" 
import torch

from monai.transforms import (
    AsDiscrete,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    RandScaleIntensityd,
    ScaleIntensityRanged,
    NormalizeIntensityd,
    Spacingd,
    RandRotate90d,
    EnsureTyped,
)

def get_monai_transformations(
        spatial_size: int,
    ):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"], ensure_channel_first = True),
            NormalizeIntensityd(keys=["image"]),
            ScaleIntensityRanged(
                keys=["label"],
                a_min=0,
                a_max=255,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller = False),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            EnsureTyped(keys=["image", "label"], device=device, track_meta=False),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(
                    spatial_size, 
                    spatial_size, 
                    spatial_size
                ),
                pos=1,
                neg=1,
                num_samples=1,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[0],
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[1],
                prob=0.5,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[2],
                prob=0.5,
            ),
            RandRotate90d(
                keys=["image", "label"],
                prob=0.5,
                max_k=3,
            ),
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.1,
                prob=1.0,
            ),
            RandScaleIntensityd(
                keys=["image"],
                factors=0.1,
                prob=1.0,
            )
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"], ensure_channel_first = True),
            NormalizeIntensityd(keys=["image"]),
            ScaleIntensityRanged(
                keys=["label"],
                a_min=0,
                a_max=255,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller = False),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            EnsureTyped(keys=["image", "label"], device=device, track_meta=True),
        ]
    )

    test_transforms = Compose(
        [
            LoadImaged(keys=["image"], ensure_channel_first = True), #channel_dim=None),
            NormalizeIntensityd(keys=["image"]),
            CropForegroundd(keys=["image"], source_key="image", allow_smaller = False),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(
                keys=["image"],
                pixdim=(1.0, 1.5, 2.0),
                mode=("bilinear"),
            ),
            EnsureTyped(keys=["image"], device=device, track_meta=True),
        ]
    )
    return train_transforms, val_transforms, test_transforms
