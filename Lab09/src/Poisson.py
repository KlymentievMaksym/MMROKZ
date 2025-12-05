import os
import cv2


class Poisson:
    def __init__(self, image_path, target_path, mask_path, offset):
        self.image_path = image_path
        self.target_path = target_path
        self.mask_path = mask_path
        self.offset = offset

        self.image = cv2.imread(image_path)
        self.target = cv2.resize(cv2.imread(target_path), self.image.shape[:2][::-1])
        self.mask = cv2.resize(cv2.imread(mask_path), self.image.shape[:2][::-1])

    def run(self):
        poisson_image = cv2.seamlessClone(self.target, self.image, self.mask, self.offset, cv2.NORMAL_CLONE)
        poisson_image_path = os.path.join(os.path.dirname(self.image_path), "poisson.png")
        cv2.imwrite(poisson_image_path, poisson_image)
        return poisson_image_path