import cv2

import os
import sys

import numpy as np


class MaskCreator:
    def __init__(self, target_path: str):
        self.target_path = target_path

        self.target = cv2.imread(target_path)
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

        if event == cv2.EVENT_LBUTTONDOWN:
            self.draw = True
            cv2.circle(self.mask, (x, y), self.brush_size, (255, 255, 255), -1)
            cv2.circle(self.target, (x, y), self.brush_size, (255, 255, 255), -1)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.draw:
                cv2.circle(self.mask, (x, y), self.brush_size, (255, 255, 255), -1)
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

        self.mask = cv2.resize(cv2.imread(mask_path), true_size)

        self._image = self.image.copy()
        self._mask = self.mask.copy()

        self.window_name = "Move mask. s:save; r:reset; q:quit"

        self.move = False
        self.offset_x = 0
        self.offset_y = 0
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
                
                self.offset_x += dx
                self.offset_y += dy
                
                self.prev_x = x
                self.prev_y = y
        elif event == cv2.EVENT_LBUTTONUP: self.move = False


    def move_mask(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 800, 600)
        cv2.setMouseCallback(self.window_name, self._move_mask)



        height, width = self.image.shape[:2]
        base_x = width // 2
        base_y = height // 2

        while True:
            # display = np.where(self.mask > 0, self.mask, self.image)
            display = self.image.copy()

            # 2. Calculate current top-left corner
            cur_x = base_x + self.offset_x
            cur_y = base_y + self.offset_y

            # 3. Handle Boundaries (Clipping) to prevent crash if moved off-screen
            # Determine intersection rectangle
            x1 = max(0, cur_x)
            y1 = max(0, cur_y)
            x2 = min(width, cur_x)
            y2 = min(height, cur_y)

            # Determine corresponding coordinates in the Target/Mask image
            tx1 = x1 - cur_x
            ty1 = y1 - cur_y
            tx2 = tx1 + (x2 - x1)
            ty2 = ty1 + (y2 - y1)

            # 4. Draw the overlay if still on screen
            if x2 > x1 and y2 > y1:
                # Extract the region of interest (ROI) from background
                roi = display[y1:y2, x1:x2]
                
                # Extract the corresponding mask and target area
                mask_slice = self.mask[ty1:ty2, tx1:tx2]
                target_slice = self.target[ty1:ty2, tx1:tx2]
                
                # Create visual blend (just for UI, not final result)
                # Where mask is white, show target; otherwise show background ROI
                # Note: mask needs to be 3-channel for this operations
                mask_3ch = mask_slice if len(mask_slice.shape) == 3 else cv2.cvtColor(mask_slice, cv2.COLOR_GRAY2BGR)
                mask_bool = mask_3ch > 0
                
                np.putmask(roi, mask_bool, target_slice)
            cv2.imshow(self.window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"): sys.exit()
            elif key == ord("s"): break
            elif key == ord("r"):
                self.image = self._image.copy()
                self.mask = self._mask.copy()
        cv2.destroyAllWindows()

        self.mask_path = os.path.join(os.path.dirname(self.target_path), "mask.png")
        cv2.imwrite(self.mask_path, self.mask)
        return self.mask_path