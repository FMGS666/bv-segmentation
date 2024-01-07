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
    DivisblePadd
)

def get_monai_transformations(
        spatial_size,
        device,
        left_pad = 62,
        right_pad = 62
    ):
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"], ensure_channel_first = True),
            #NormalizeIntensityd(keys=["image"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=0,
                a_max=2**16,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            ScaleIntensityRanged(
                keys=["label"],
                a_min=0,
                a_max=255,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            Spacingd(
                keys=["image", "label"],
                pixdim=(2.0, 2.0, 2.0),
                mode=("bilinear", "nearest"),
            ),
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
            #NormalizeIntensityd(keys=["image"]),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=0,
                a_max=2**16,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
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
            Spacingd(
                keys=["image", "label"],
                pixdim=(2.0, 2.0, 2.0),
                mode=("bilinear", "nearest"),
            ),
            EnsureTyped(keys=["image", "label"], device=device, track_meta=True),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(
                    2*spatial_size, 
                    2*spatial_size, 
                    2*spatial_size
                ),
                pos=1,
                neg=1,
                num_samples=1,
                image_key="image",
                image_threshold=0,
                allow_smaller=True
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
    test_transforms = Compose(
        [
            LoadImaged(keys=["image"], ensure_channel_first = True), 
            #NormalizeIntensityd(keys=["image"]),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=0,
                a_max=2**16,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            Orientationd(keys=["image"], axcodes="RAS"),
            EnsureTyped(keys=["image"], device=device, track_meta=True),
            DivisblePadd(keys= ["image","label"],k=patch_size, allow_missing_keys = True)
        ]
    )
    return train_transforms, val_transforms, test_transforms
