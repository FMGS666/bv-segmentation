import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm

from ..file_utils.tif_file_loader import TifFileLoader
from ..file_utils.tif_iterable_folder import Tif3DVolumeIterableFolder

def create_volume_animations(
        dataset_folder: Tif3DVolumeIterableFolder
    ) -> None:
    frames = []
    fig = plt.figure()
    fig, (image_axes, mask_axes) = plt.subplots(
        nrows = 1, 
        ncols = 2
    )
    for slice_image_id, slice_image_path, slice_mask_path in tqdm(dataset_folder):
        slice_image = TifFileLoader(
            slice_image_path
        )
        slice_mask = TifFileLoader(
            slice_mask_path
        )
        frames.append(
            [
                image_axes.imshow(slice_image.image,animated=True),
                mask_axes.imshow(slice_mask.image, cmap=cm.Greys_r,animated=True),
            ]
        )
    ani = animation.ArtistAnimation(fig, frames, interval=100, blit=True,
                                    repeat_delay=1000)
    ani.save(f'animations/{dataset.name}.mp4')
