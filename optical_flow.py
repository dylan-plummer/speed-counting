import cv2
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


def open_video(file):
    cap = cv2.VideoCapture(file)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))

    fc = 0
    ret = 1

    while True:
        try:
            ret, buf[fc] = cap.read()
            fc += 1
        except:
            print('Done reading video')
            break

    cap.release()

    return buf


def high_pass_filter(img, dft_shift, filter=30):
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    fshift = np.copy(dft_shift)
    fshift[crow - filter:crow + filter, ccol - filter:ccol + filter] = 0

    # inverse DFT
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    mag, ang = cv2.cartToPolar(img_back[:, :, 0], img_back[:, :, 1])
    return mag, ang


def low_pass_filter(img, dft_shift, filter=30):
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2

    # create a mask first, center square is 1, remaining all zeros
    mask = np.zeros((rows, cols, 2), np.uint8)
    mask[crow - filter:crow + filter, ccol - filter:ccol + filter] = 1

    # apply mask and inverse DFT
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    mag, ang = cv2.cartToPolar(img_back[:, :, 0], img_back[:, :, 1])
    return mag, ang


def fourier(video, i, j):
    prev = cv2.cvtColor(video[i], cv2.COLOR_BGR2GRAY)
    next = cv2.cvtColor(video[j], cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    dft = cv2.dft(np.float32(flow), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    hsv = np.zeros_like(video[0])
    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(dft_shift[..., 0], dft_shift[..., 1])
    '''
    plt.subplot(121), plt.imshow(mag, cmap='plasma')
    plt.title('mag'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(ang, cmap='plasma')
    plt.title('ang'), plt.xticks([]), plt.yticks([])
    plt.savefig('frames/' + str(i) + '.png')
    plt.clf()
    '''
    return flow


video = open_video('test.mp4')
print(video.shape)
signal = []

for i in range(0, len(video) - 1):
    intensity = fourier(video, i, i + 1)
    signal.append(intensity)

wavelet_transform(signal)
plt.plot(signal)
plt.show()
