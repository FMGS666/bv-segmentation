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
    GridPatchd,
    Padd,
    Pad,
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
        spatial_size,
        device
    ):
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"], ensure_channel_first = True),
            NormalizeIntensityd(keys=["image"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            EnsureTyped(keys=["image", "label"], device=device, track_meta=False),
            ScaleIntensityRanged(
                keys=["label"],
                a_min=0,
                a_max=255,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
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
                prob=0.5,
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
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(
                    3*spatial_size, 
                    3*spatial_size, 
                    3*spatial_size
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
                prob=0.5,
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
    test_padder = Pad(
        to_pad = [
            (0, 0),
            (62, 63), 
            (0, 0), 
            (0, 0)
        ]
    )
    test_transforms = Compose(
        [
            LoadImaged(keys=["image"], ensure_channel_first = True), 
            NormalizeIntensityd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            EnsureTyped(keys=["image"], device=device, track_meta=True),
            Padd(
                keys="image",
                padder = test_padder
            ),
            GridPatchd(
                keys=["image"],
                patch_size=(
                    spatial_size, 
                    spatial_size, 
                    spatial_size
                ),
                pad_mode="constant"
            )
        ]
    )
    return train_transforms, val_transforms, test_transforms
