import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


class Poisson:
    def __init__(self, image_path, target_path, mask_path, offset, target_size: tuple = (float("inf"), float("inf"))):
        self.image_path = image_path
        self.target_path = target_path
        self.mask_path = mask_path

        self.image = cv2.imread(image_path)

        max_size = np.array(target_size)
        image_size = self.image.shape[:2][::-1]
        true_size = np.where(image_size < max_size, image_size, max_size).astype(int)

        self.target = cv2.resize(cv2.imread(target_path), true_size)
        self.mask = cv2.resize(cv2.imread(mask_path), true_size)

        offset_height, offset_width = offset
        height, width = self.target.shape[:2]

        center_x = int(offset_height + (width // 2))
        center_y = int(offset_width + (height // 2))
        
        self.offset = (center_y, center_x)
        print(offset)
        print(self.offset)

    def run(self):
        poisson_image = cv2.seamlessClone(self.target, self.image, self.mask, self.offset, cv2.NORMAL_CLONE)
        poisson_image_path = os.path.join(os.path.dirname(self.image_path), "poisson.png")
        cv2.imwrite(poisson_image_path, poisson_image)
        return poisson_image_path