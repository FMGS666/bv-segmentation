import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm

from ..file_utils.tif_file_loader import TifFileLoader

def tif_image_mask_pair_distribution_plotter(
        slice_image_loader: TifFileLoader | None,
        slice_mask_loader: TifFileLoader | None,
        figsize: int = 10,
        **hist_kwargs: dict
    ) -> None:
    if slice_mask_loader and slice_image_loader:
        fig, (image_axes, mask_axes) = plt.subplots(
            nrows = 1, 
            ncols = 2,
            figsize = (figsize, figsize)
        )
        mask_axes.set_title(f"Mask pixel value distribution (id: {slice_mask_loader.slice_id})")
        mask_axes.hist(
            slice_mask_loader.image_array,
            **hist_kwargs
        )
        image_axes.set_title(f"Tomographic Image pixel value distribution(id: {slice_image_loader.slice_id})")
        image_axes.hist(
            slice_image_loader.image_array,
            **hist_kwargs
        )
    if slice_image_loader and not slice_mask_loader:
        fig, image_axes = plt.subplots(
            nrows = 1, 
            ncols = 1,
            figsize = figsize
        )
        image_axes.set_title(f"Tomographic Image pixel value distribution(id: {slice_image_loader.slice_id})")
        image_axes.hist(
            slice_image_loader.image_array,
            **hist_kwargs
        )
    if slice_mask_loader and not slice_image_loader:
        fig, mask_axes = plt.subplots(
            nrows = 1, 
            ncols = 1,
            figsize = figsize
        )
        mask_axes.set_title(f"Mask (id: {slice_mask_loader.slice_id})")
        image_axes.hist(
            slice_mask_loader.image_array,
            **hist_kwargs
        )
        image_axes.set_title(f"Tomographic (id: {slice_image_loader.slice_id})")
        image_axes.hist(
            slice_image_loader.image_array,
            **hist_kwargs
        )
    fig.tight_layout()
    plt.show()