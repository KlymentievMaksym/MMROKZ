import cv2

import os
import sys

import numpy as np


class MaskCreator:
    def __init__(self, target: cv2.Mat, target_path: str):  #, image_path
        # self.image_path = image_path
        self.target_path = target_path

        # self.image = cv2.imread(image_path)
        # self.target = cv2.imread(target_path)
        self.target = target
        # self.mask = np.zeros(self.target.shape[:-1])
        self.mask = np.zeros_like(self.target)

        self._target = self.target.copy()
        self._mask = self.mask.copy()

        self.brush_size = 4
        self.draw = False
        self.window_name = "Draw mask. s:save; r:reset; q:quit"

    def _draw_mask(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0: self.brush_size += 1
            else: self.brush_size -= 1
            self.brush_size = max(1, min(self.brush_size, 100))

        if event == cv2.EVENT_LBUTTONDOWN: self.draw = True
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.draw:
                cv2.circle(self.mask, (x, y), self.brush_size, (255, 255, 255), -1)
                cv2.circle(self.target, (x, y), self.brush_size, (255, 255, 255), -1)
        elif event == cv2.EVENT_LBUTTONUP: self.draw = False

    def draw_mask(self, save: bool = False):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 800, 600)
        cv2.setMouseCallback(self.window_name, self._draw_mask)

        while True:
            display = self.target.copy()
            cv2.putText(display, f"Brush: {self.brush_size} px", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow(self.window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"): sys.exit()
            elif key == ord("s"): break
            elif key == ord("r"):
                self.target = self._target.copy()
                self.mask = self._mask.copy()
        cv2.destroyAllWindows()

        if save:
            self.mask_path = os.path.join(os.path.dirname(self.target_path), "mask.png")
            cv2.imwrite(self.mask_path, self.mask)
        return self.mask


class MaskMover:
    def __init__(self, image: cv2.Mat, mask: cv2.Mat):
        self.image = image
        self.mask = mask  # / 255 if mask.dtype == int else mask

        self._image = self.image.copy()
        self._mask = self.mask.copy()

        self.move = False
        self.window_name = "Move mask. s:save; r:reset; q:quit"

    def _move_mask(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN: self.move = True
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.move:
                pass
        elif event == cv2.EVENT_LBUTTONUP: self.move = False


    def move_mask(self, save: bool = False):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 800, 600)
        cv2.setMouseCallback(self.window_name, self._move_mask)

        while True:
            display = np.where(self.mask > 0, self.mask, self.image)
            cv2.imshow(self.window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"): sys.exit()
            elif key == ord("s"): break
            elif key == ord("r"):
                self.target = self._target.copy()
                self.mask = self._mask.copy()
        cv2.destroyAllWindows()

        # if save:
        #     self.mask_path = os.path.join(os.path.dirname(self.target_path), "mask.png")
        #     cv2.imwrite(self.mask_path, self.mask)
        return self.mask