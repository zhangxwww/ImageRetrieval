import os
from PIL import Image
import numpy as np


class Dataset:
    def __init__(self, path='./Dataset',
                 all_list='AllImages.txt',
                 query_list='QueryImages.txt'):
        self.path = path
        self.all_list = all_list
        self.query_list = query_list

    def get_all_images(self):
        return self._read_images(self.all_list)

    def get_query_images(self):
        return self._read_images(self.query_list)

    def _read_images(self, image_list):
        with open(image_list) as f:
            lines = f.readlines()
        lines = filter(lambda x: 'jpg' in x, lines)
        image_names = list(map(lambda x: x.split()[0], lines))
        all_images = list(map(lambda x: np.array(Image.open(os.path.join(self.path, x))), image_names))
        return all_images, image_names
