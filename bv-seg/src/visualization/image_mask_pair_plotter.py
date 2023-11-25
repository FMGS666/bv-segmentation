import matplotlib.pyplot as plt

from ..file_utils.tif_file_loader import TifFileLoader

def tif_image_mask_pair_plotter(
        slice_image_loader: TifFileLoader | None,
        slice_mask_loader: TifFileLoader | None,
        figsize: int = 20
    ) -> None:
    if slice_mask_loader and slice_image_loader:
        fig, (image_axes, mask_axes) = plt.subplots(
            nrows = 1, 
            ncols = 2,
            figsize = (figsize, figsize)
        )
        mask_axes.set_title(f"Mask (id: {slice_mask_loader.slice_id})")
        mask_axes.imshow(
            slice_mask_loader.image
        )
        image_axes.set_title(f"Tomographic Image (id: {slice_image_loader.slice_id})")
        image_axes.imshow(
            slice_image_loader.image
        )
    if slice_image_loader and not slice_mask_loader:
        fig, image_axes = plt.subplots(
            nrows = 1, 
            ncols = 1,
            figsize = figsize
        )
        image_axes.set_title(f"Tomographic Image (id: {slice_image_loader.slice_id})")
        image_axes.imshow(
            slice_image_loader.image
        )
    if slice_mask_loader and not slice_image_loader:
        fig, mask_axes = plt.subplots(
            nrows = 1, 
            ncols = 1,
            figsize = figsize
        )
        mask_axes.set_title(f"Mask (id: {slice_mask_loader.slice_id})")
        image_axes.imshow(
            slice_mask_loader.image
        )
        image_axes.set_title(f"Tomographic Image (id: {slice_image_loader.slice_id})")
        image_axes.imshow(
            slice_image_loader.image
        )
    fig.tight_layout()
