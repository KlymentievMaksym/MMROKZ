import cv2
import numpy as np
import matplotlib.pyplot as plt

img_path = "./Lab09/images/ball.jpg" 
original = cv2.imread(img_path)
original = cv2.resize(original, (600, 400))

x, y, w, h  = cv2.selectROI("Select the area to be corrupted", original)

top, bottom, left, right = y, y + h, x, x + w

cv2.destroyAllWindows()

# original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
corrupted = original.copy()

corrupted[top:bottom, left:right] = np.median(corrupted[top:bottom, left:right], axis=(0, 1))

mask = np.zeros(original.shape[:2], dtype=np.uint8)
mask[top:bottom, left:right] = 255

center = (int((left + right) / 2), int((top + bottom) / 2))


fig, axes = plt.subplots(1, 4, figsize=(15, 5))
axes[0].set_title("[Original]")
axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))

axes[1].set_title("[Corrupted]")
axes[1].imshow(cv2.cvtColor(corrupted, cv2.COLOR_BGR2RGB))

axes[2].set_title("[Mask]")
axes[2].imshow(mask, cmap="gray")

restored = cv2.seamlessClone(original.copy(), corrupted.copy(), mask.copy(), center, cv2.NORMAL_CLONE)
axes[3].set_title("[Restored]")
axes[3].imshow(cv2.cvtColor(restored, cv2.COLOR_BGR2RGB))
plt.show()

cv2.imwrite("./Lab09/images/corrupted.jpg", corrupted)
cv2.imwrite("./Lab09/images/corrupted_mask.png", mask)
cv2.imwrite("./Lab09/images/restored.jpg", restored)

cv2.waitKey(0)
cv2.destroyAllWindows()