import cv2

import os
import sys

import numpy as np


class MaskCreator:
    def __init__(self, target_path: str):
        self.target_path = target_path

        self.target = cv2.imread(target_path)
        self.mask = np.zeros(self.target.shape[:2], dtype=np.uint8)
        # self.mask = np.zeros_like(self.target)

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

        if event == cv2.EVENT_LBUTTONDOWN:
            self.draw = True
            cv2.circle(self.mask, (x, y), self.brush_size, 255, -1)
            cv2.circle(self.target, (x, y), self.brush_size, (255, 255, 255), -1)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.draw:
                cv2.circle(self.mask, (x, y), self.brush_size, 255, -1)
                cv2.circle(self.target, (x, y), self.brush_size, (255, 255, 255), -1)
        elif event == cv2.EVENT_LBUTTONUP: self.draw = False

    def draw_mask(self):  #, save: bool = False
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

        # if save:
        self.mask_path = os.path.join(os.path.dirname(self.target_path), "mask.png")
        cv2.imwrite(self.mask_path, self.mask)
        return self.mask_path


class MaskMover:
    def __init__(self, image_path: str, mask_path: str, target_size: tuple = (float("inf"), float("inf"))):
        self.image_path = image_path
        self.mask_path = mask_path

        self.image = cv2.imread(image_path)
        
        max_size = np.array(target_size)
        image_size = self.image.shape[:2][::-1]
        true_size = np.where(image_size < max_size, image_size, max_size).astype(int)

        self.width, self.height = true_size

        self.mask = cv2.resize(cv2.imread(mask_path), true_size)

        self._image = self.image.copy()
        self._mask = self.mask.copy()

        self.window_name = "Move mask. s:save; r:reset; q:quit"

        self.move = False
        self.offset_width = 0
        self.offset_height = 0
        self.prev_x = 0
        self.prev_y = 0


    def _move_mask(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.move = True
            self.prev_x = x
            self.prev_y = y
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.move:
                dx = x - self.prev_x
                dy = y - self.prev_y
                
                self.offset_width += dx
                self.offset_height += dy
                
                self.prev_x = x
                self.prev_y = y
        elif event == cv2.EVENT_LBUTTONUP: self.move = False


    def move_mask(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 800, 600)
        cv2.setMouseCallback(self.window_name, self._move_mask)

        while True:
            display = self.image.copy()
            
            x1 = max(self.offset_width, 0)
            y1 = max(self.offset_height, 0)
            x2 = min(self.offset_width + self.width, self.image.shape[1])
            y2 = min(self.offset_height + self.height, self.image.shape[0])

            mask_x1 = x1 - self.offset_width
            mask_y1 = y1 - self.offset_height
            mask_x2 = mask_x1 + (x2 - x1)
            mask_y2 = mask_y1 + (y2 - y1)

            if x2 > x1 and y2 > y1:
                roi = display[y1:y2, x1:x2]
                
                mask_chunk = self.mask[mask_y1:mask_y2, mask_x1:mask_x2]

                roi[:] = np.where(mask_chunk > 0, mask_chunk, roi)

            cv2.imshow(self.window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"): sys.exit()
            elif key == ord("s"): break
            elif key == ord("r"):
                self.mask = self._mask.copy()
        cv2.destroyAllWindows()

        offset = (self.offset_height, self.offset_width)
        print(offset)
        return offset
