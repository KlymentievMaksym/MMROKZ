from src import MaskCreator, MaskMover, Poisson
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def main(image_path: str, target_path: str, mask_path: str = None, offset: tuple = None, target_size: tuple = (float("inf"), float("inf"))):
    mask_path = MaskCreator(target_path).draw_mask() if mask_path is None else mask_path
    offset = MaskMover(image_path, mask_path, target_size).move_mask() if offset is None else offset

    poison_image_path = Poisson(image_path, target_path, mask_path, offset, target_size).run()
    cv2.namedWindow("Poisson", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Poisson", 800, 600)
    cv2.imshow("Poisson", cv2.imread(poison_image_path))
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == "__main__":
    # main("./Lab09/images/Towers.jpg", "./Lab09/images/Airplane.jpg", target_size=(350, 250))
    main("./Lab09/images/water.jpg", "./Lab09/images/ball.jpg")
    # main("./Lab09/images/Towers.jpg", "./Lab09/images/Airplane.jpg", mask_path="./Lab09/images/mask.png", target_size=(350, 250))
    # main("./Lab09/images/water.jpg", "./Lab09/images/ball.jpg", mask_path="./Lab09/images/mask.png")