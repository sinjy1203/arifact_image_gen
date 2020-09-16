## import package
import numpy as np
import matplotlib.pyplot as plt

## load image
img = plt.imread("Lenna.png")

# gray image generation
# img = np.mean(img, axis=2, keepdims=True)

img_size = img.shape

cmap = "gray" if img_size[2] == 1 else None

plt.imshow(np.squeeze(img), cmap=cmap, vmin=0, vmax=1)
plt.title("Ground Truth")
plt.show()

## uniform sampling
ds_y = 2
ds_x = 4

msk = np.zeros(img_size)
msk[::ds_y, ::ds_x, :] = 1

dst = img*msk

plt.subplot(131)
plt.imshow(np.squeeze(img), cmap=cmap, vmin=0, vmax=1)
plt.title("Ground Truth")

plt.subplot(132)
plt.imshow(np.squeeze(msk), cmap=cmap, vmin=0, vmax=1)
plt.title("Uniform sampling mask")

plt.subplot(133)
plt.imshow(np.squeeze(dst), cmap=cmap, vmin=0, vmax=1)
plt.title("sampling image")

plt.show()

## random sampling
rnd = np.random.rand(img_size[0], img_size[1], img_size[2])

prob = 0.5

msk = (rnd > prob).astype(np.float)

dst = img * msk

plt.subplot(131)
plt.imshow(np.squeeze(img), cmap=cmap, vmin=0, vmax=1)
plt.title("Ground Truth")

plt.subplot(132)
plt.imshow(np.squeeze(msk), cmap=cmap, vmin=0, vmax=1)
plt.title("random sampling mask")

plt.subplot(133)
plt.imshow(np.squeeze(dst), cmap=cmap, vmin=0, vmax=1)
plt.title("sampling image")

plt.show()