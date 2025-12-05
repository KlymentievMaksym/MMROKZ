import os
import cv2


class Poisson:
    def __init__(self, image, target, mask, offset):
        self.image = image
        self.target = target
        self.mask = mask
        self.offset = ((self.image.shape[1]) // 2, (self.image.shape[0]) // 2)

    def run(self):
        print(f"[TargetShape]: {self.target.shape}\n[ImageShape]: {self.image.shape}\n[MaskShape]: {self.mask.shape}\n[Offset]: {self.offset}")
        print(f"[Mask]: {self.mask.max()}")
        poisson_image = cv2.seamlessClone(self.target, self.image, self.mask, self.offset, cv2.NORMAL_CLONE)
        # poisson_image_path = os.path.join(os.path.dirname(self.image_path), "poisson.png")
        # cv2.imwrite(poisson_image_path, poisson_image)
        # return poisson_image_path
        return poisson_image