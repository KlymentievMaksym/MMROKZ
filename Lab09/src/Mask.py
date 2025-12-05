import cv2

import os

import numpy as np


class Mask:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        self.mask = np.zeros_like(self.image)

        self._image = self.image.copy()
        self._mask = self.mask.copy()

        self.brush_size = 4
        self.draw = False
        self.window_name = "Draw mask. s:save; r:reset; q:quit"

    def draw_mask(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:
                self.brush_size += 2
            else:
                self.brush_size -= 2

            self.brush_size = max(1, min(self.brush_size, 100))

        if event == cv2.EVENT_LBUTTONDOWN:
            self.draw = True
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.draw:
                cv2.circle(self.mask, (x, y), self.brush_size, (255, 255, 255), -1)
                cv2.circle(self.image, (x, y), self.brush_size, (255, 255, 255), -1)
        elif event == cv2.EVENT_LBUTTONUP:
            self.draw = False

    def run(self):
        self.mask_path = os.path.join(os.path.dirname(self.image_path), "mask.png")
        # if os.path.exists(self.mask_path):
        #     return self.mask_path

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1200, 800)
        cv2.setMouseCallback(self.window_name, self.draw_mask)

        while True:
            display = self.image.copy()

            cv2.putText(display, f"Brush: {self.brush_size} px", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow(self.window_name, display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                self._mask = self.mask.copy()
                break
            elif key == ord("r"):
                self.image = self._image.copy()
                self.mask = self._mask.copy()

        cv2.imwrite(self.mask_path, self.mask)
        cv2.destroyAllWindows()
        return self.mask_path
