from src import Mask, Poisson
import os
import cv2


def main(image_path: str, target_path: str):
    mask_path = Mask(target_path).run()
    # mask_path = Mask(image_path).run()
    poison_image_path = Poisson(image_path, target_path, mask_path, (-1, -1)).run()
    cv2.imshow("Poisson", cv2.imread(poison_image_path))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # os.remove(mask_path)
    # os.remove(poison_image_path)


if __name__ == "__main__":
    # main("./Lab09/images/Towers.jpg", "./Lab09/images/Airplane.jpg")
    main("./Lab09/images/Airplane.jpg", "./Lab09/images/Towers.jpg")
