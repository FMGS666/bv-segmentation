f"""

"""
import numpy as np
import os
import re

from pathlib import Path
from PIL import Image
from monai.data.image_reader import ImageReader

from .tif_file_loader import TifFileLoader

class MonAiImageReader(ImageReader):
    def __init__(
            self
        ) -> None:
        super(MonAiImageReader, self).__init__()
    
    def read(
            self,
            file_name: str | Path
        ) -> Image:
        img = []
        if os.path.isfile(file_name):
            file_loader = TifFileLoader(file_name)
            image_array = np.rollaxis(file_loader.image_array, 0, 3)
            img.append(image_array)
        else:
            images = os.listdir(file_name)
            for image in images:
                image_path = os.path.join(file_name, image)
                file_loader = TifFileLoader(file_name)
                image_array = np.rollaxis(file_loader.image_array, 0, 3)
                img.append(image_array)
        return img if len(img) > 1 else img[0]

    def get_data(
            self,
            file_name: str | Path
        ) -> tuple[np.ndarray, dict] | list[tuple[np.ndarray, dict]]:
        img = []
        if os.path.isfile(file_name):
            file_loader = TifFileLoader(file_name)
            image_array = np.rollaxis(file_loader.image_array, 0, 3)
            data_tuple = (
                image_array,
                {
                    "slice_id": file_loader.slice_id,
                    "dataset": re.findall("/(kidney.*)/", file_name)[0]
                }
            )
            img.append(data_tuple)
        else:
            images = os.listdir(file_name)
            for image in images:
                image_path = os.path.join(file_name, image)
                file_loader = TifFileLoader(image_path)
                image_array = np.rollaxis(file_loader.image_array, 0, 3)
                data_tuple = (
                    image_array,
                    {
                        "slice_id": file_loader.slice_id,
                        "dataset": re.findall("/(kidney.*)/", file_name)[0]
                    }
                )
                img.append(data_tuple)
        return img if len(img) > 1 else img[0]
    
    def verify_suffix(
            self,
            file_name
        ) -> bool:
        return True if len(re.findall("\.tif", file_name)) > 0 else False

    
            
