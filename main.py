import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

import seaborn as sns

sns.set()
sns.set_style('dark')


# Current timestamp
def current_date_time():
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


current_date_time()


################################################################

def apply_fourier_filter(img, mask):
    mask = mask[:, :, np.newaxis]
    img_fourier = cv2.dft(img.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
    img_fourier_shift = np.fft.fftshift(img_fourier)
    img_fourier_shift *= mask
    img_fourier_shift_back = np.fft.ifftshift(img_fourier_shift)
    img_fourier_inverse = cv2.idft(img_fourier_shift_back, flags=cv2.DFT_SCALE)

    return img_fourier_inverse


def fourier_analysis(img):
    fourier_img = cv2.dft(img.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
    fourier_img_shift = np.fft.fftshift(fourier_img)
    real = fourier_img_shift[:, :, 0]
    imag = fourier_img_shift[:, :, 1]
    magnitude = cv2.magnitude(real, imag)
    phase = cv2.phase(real, imag)
    return magnitude, phase


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def disp(img, title='', s=8, vmin=None, vmax=None):
    plt.figure(figsize=(s, s))
    plt.axis('off')
    if vmin is not None and vmax is not None:
        plt.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.show()
