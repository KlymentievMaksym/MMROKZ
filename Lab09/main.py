from src import MaskCreator, MaskMover, Poisson
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def main(image_path: str, target_path: str, mask_path: str = None, offset: tuple = None, target_size: tuple = (float("inf"), float("inf"))):
    # image = cv2.imread(image_path)
    # target = cv2.imread(target_path)

    mask_path = MaskCreator(target_path).draw_mask() if mask_path is None else mask_path
    offset = MaskMover(image_path, mask_path).move_mask() if offset is None else offset

    poison_image_path = Poisson(image_path, target_path, mask_path, offset, target_size).run()
    cv2.imshow("Poisson", cv2.imread(poison_image_path))
    # cv2.imshow("Poisson", poison_image_path)
    # cv2.imshow("Poisson", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # os.remove(mask_path)
    # os.remove(poison_image_path)


if __name__ == "__main__":
    # mask2, poisson2 = main("./Lab09/images/Towers.jpg", "./Lab09/images/Airplane.jpg", target_size=(350, 250))
    main("./Lab09/images/Towers.jpg", "./Lab09/images/Airplane.jpg", mask_path="./Lab09/images/mask.png", target_size=(350, 250))
    # print(np.where(mask1!=mask2))
    # plt.subplot(121)
    # plt.imshow(mask1)
    # plt.subplot(122)
    # plt.imshow(mask2)
    # plt.show()
    # plt.subplot(121)
    # plt.imshow(poisson1)
    # plt.subplot(122)
    # plt.imshow(poisson2)
    # plt.show()