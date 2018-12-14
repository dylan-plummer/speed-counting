import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from wavelets_pytorch.wavelets import Morlet, Ricker, DOG
from examples.plot import plot_scalogram
from wavelets_pytorch.transform import WaveletTransform        # SciPy version

"""
Example script to plot SciPy and PyTorch implementation outputs side-to-side.
"""

MHI_DURATION = 0.5
DEFAULT_THRESHOLD = 32
MAX_TIME_DELTA = 0.25
MIN_TIME_DELTA = 0.05

fps = 20
dt  = 0.01
dj  = 0.125
unbias = False
batch_size = 32
wavelet = Morlet(w0=6)

t_min = 1
t_max = 10


def divergence(field):
    "return the divergence of a n-D field"
    return np.sum(np.gradient(field),axis=1)


def wavelet_transform(batch, dt=dt, dj=dj, wavelet=wavelet, unbias=unbias):
    ######################################
    # Performing wavelet transform

    print('Batch', batch.shape)
    wa = WaveletTransform(dt, dj, wavelet, unbias=unbias)

    power = wa.power(batch)

    ######################################
    # Plotting

    fig, ax = plt.subplots(1, 3, figsize=(12,3))
    ax = ax.flatten()

    t = np.arange(len(batch[0]))
    ax[0].plot(t, batch[0])
    ax[0].set_title(r'$f(t) = \sin(2\pi \cdot f t) + \mathcal{N}(\mu,\,\sigma^{2})$')
    ax[0].set_xlabel('Time (s)')

    # Plot scalogram for SciPy implementation
    plot_scalogram(power[0], wa.fourier_periods, t, ax=ax[1], scale_legend=False)
    ax[1].axhline(1.0 / t[0], lw=1, color='k')
    ax[1].set_title('Scalogram (SciPy)'.format(1.0/t[0]))

    plt.tight_layout()
    plt.show()