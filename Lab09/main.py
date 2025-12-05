from src import MaskCreator, MaskMover, Poisson
import os
import cv2
import numpy as np


def main(image_path: str, target_path: str, mask_path: str = None, target_size: tuple = (float("inf"), float("inf"))):
    image = cv2.imread(image_path)
    target = cv2.imread(target_path)

    max_size = np.array(target_size)
    image_size = image.shape[:2][::-1]
    true_size = np.where(image_size < max_size, image_size, max_size).astype(int)
    target = cv2.resize(target, true_size)

    print(true_size)
    mask = MaskCreator(target, target_path).draw_mask(save=False) if mask_path is None else cv2.resize(cv2.imread(mask_path), true_size)
    # offset = MaskMover(image, mask)

    poison_image_path = Poisson(image, target, mask, (0, 0)).run()
    # cv2.imshow("Poisson", cv2.imread(poison_image_path))
    cv2.imshow("Poisson", poison_image_path)
    # cv2.imshow("Poisson", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # os.remove(mask_path)
    # os.remove(poison_image_path)


if __name__ == "__main__":
    main("./Lab09/images/Towers.jpg", "./Lab09/images/Airplane.jpg", mask_path="./Lab09/images/mask.png", target_size=(350, 250))
    main("./Lab09/images/Towers.jpg", "./Lab09/images/Airplane.jpg", target_size=(350, 250))
