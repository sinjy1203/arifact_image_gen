## import package
import numpy as np
import matplotlib.pyplot as plt

from skimage.transform import rescale

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
# 흑백
# rnd = np.random.rand(img_size[0], img_size[1], 1)

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

## gaussian sampling
ly = np.linspace(-1, 1, img_size[0])
lx = np.linspace(-1, 1, img_size[1])

x, y = np.meshgrid(lx, ly)

x0 = 0
y0 = 0
sgmx = 1
sgmy = 1

a = 1

gaus = a * np.exp(-((x - x0)**2 / 2*sgmx**2 + (y - y0)**2 / 2*sgmy**2))
gaus = np.tile(gaus[:, :, np.newaxis], (1, 1, img_size[2]))

rnd = np.random.rand(img_size[0], img_size[1], img_size[2])
msk = (rnd < gaus).astype(np.float)
# 흑백
# rnd = np.random.rand(img_size[0], img_size[1], 1)
# rnd = np.tile(rnd, (1,1,img_size[2]))
# msk = (rnd < gaus).astype(np.float)

dst = img*msk

plt.subplot(131)
plt.imshow(np.squeeze(img), cmap=cmap, vmin=0, vmax=1)
plt.title("Grond Truth")

plt.subplot(132)
plt.imshow(np.squeeze(msk), cmap=cmap, vmin=0, vmax=1)
plt.title("Gaussian sampling mask")

plt.subplot(133)
plt.imshow(np.squeeze(dst), cmap=cmap, vmin=0, vmax=1)
plt.title("sampling image")

plt.show()

## random noise
sgm = 60.0

noise = sgm / 255.0 * np.random.randn(img_size[0], img_size[1], img_size[2])

dst = img + noise

plt.subplot(131)
plt.imshow(np.squeeze(img), cmap=cmap, vmin=0, vmax=1)
plt.title("Ground Truth")

plt.subplot(132)
plt.imshow(np.squeeze(noise), cmap=cmap, vmin=0, vmax=1)
plt.title("Noise")

plt.subplot(133)
plt.imshow(np.squeeze(dst), cmap=cmap, vmin=0, vmax=1)
plt.title("Noisy image")

plt.show()

## super resolution
dw = 1 / 5.0
order = 1

dst_dw = rescale(img, scale=(dw, dw, 1), order=order)
dst_up = rescale(dst_dw, scale=(1/dw, 1/dw, 1), order=order)

plt.subplot(131)
plt.imshow(np.squeeze(img), cmap=cmap, vmin=0, vmax=1)
plt.title("Ground Truth")

plt.subplot(132)
plt.imshow(np.squeeze(dst_dw), cmap=cmap, vmin=0, vmax=1)
plt.title("downscaled image")

plt.subplot(133)
plt.imshow(np.squeeze(dst_up), cmap=cmap, vmin=0, vmax=1)
plt.title("upscaled image")

plt.show()