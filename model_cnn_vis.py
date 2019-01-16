from itertools import product

import math
import numpy as np
import cv2
from Bio import pairwise2
import pandas as pd
import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from keras.models import model_from_json, Model
from keras.utils import np_utils

data_dir = os.getcwd() + '/data/'
kernel_size = 16
kernel_frames = 32
frame_size = 64
window_size = 128


def video_to_flow_field(video):
    flow = np.array([])
    for i in range(len(video) - 1):
        field = get_flow_field(video, i, i + 1)
        flow = np.append(flow, field)
    return np.reshape(flow, (video.shape[0] - 1, video.shape[1], video.shape[2], 2))


def open_video(file, window_size):
    print('\nOpening', file)
    cap = cv2.VideoCapture(file)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = frame_size
    frameHeight = frame_size

    buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))

    fc = 0
    ret = 1

    while True:
        try:
            ret, buf[fc] = cv2.resize(cap.read(), (frame_size, frame_size))
            fc += 1
        except:
            # print('Done reading video')
            break
    print('Done reading video')
    cap.release()
    return buf


def get_flow_field(video, i, j):
    prev = cv2.cvtColor(video[i], cv2.COLOR_BGR2GRAY)
    next = cv2.cvtColor(video[j], cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros_like(video[i])
    hsv[..., 1] = 255
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return flow

def plot_weights(weights):
    print(weights.shape)
    for frame_i in range(kernel_frames):
        fig, axes = plt.subplots(8, 8)
        print('Saving frame', frame_i)
        for i in range(len(weights)):
            r = i // 8
            c = i % 8
            x, y = np.meshgrid(np.arange(kernel_size), np.arange(kernel_size))
            u = weights[i][frame_i][..., 0]
            v = weights[i][frame_i][..., 1]
            mag, ang = cv2.cartToPolar(u, v)
            hsv = np.zeros((kernel_size, kernel_size, 3))
            hsv[..., 1] = 255
            hsv[..., 0] = ang * 180 // np.pi // 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            bgr = cv2.cvtColor(np.uint8(hsv), cv2.COLOR_HSV2RGB_FULL)
            axes[r][c].imshow(bgr)
            #axes[r][c].quiver(x, y, u, v, mag, edgecolor='k', width=0.05, pivot='tip')
            axes[r][c].set_xticks([])
            axes[r][c].set_yticks([])
        plt.tight_layout(0.5)
        plt.savefig('saved/' + str(frame_i) + '.png', dpi=250)
        plt.clf()


def plot_conv_layer():
    # load data

    dir = 'models/'
    with open(dir + 'model_architecture.json', 'r') as f:
        model = model_from_json(f.read())
    # Load weights into the new model
    model.load_weights(dir + 'model_weights.h5')
    layer_outputs = [layer.output for layer in model.layers]
    print(layer_outputs)
    for layer in layer_outputs:
        print(layer)
    conv_embds = np.array(model.layers[1].get_weights())
    print(conv_embds[0].shape)
    weights = np.reshape(conv_embds[0], (64, kernel_frames, kernel_size, kernel_size, 2))
    print(weights)
    plot_weights(weights)


def predict_test():
    dir = 'models/'
    with open(dir + 'model_architecture.json', 'r') as f:
        model = model_from_json(f.read())
    # Load weights into the new model
    model.load_weights(dir + 'model_weights.h5')
    video_path = np.random.choice(os.listdir(data_dir + 'videos'))
    video = open_video(data_dir + '/videos/' + video_path, window_size=window_size)
    num_clips = 0
    for start_frame in range(0, len(video), window_size):
        if start_frame + window_size < len(video):
            clip = video[start_frame:start_frame + window_size]
            flow_field = video_to_flow_field(clip)
            pred = model.predict(np.array([flow_field]))
            print(pred)


predict_test()
#plot_conv_layer()
