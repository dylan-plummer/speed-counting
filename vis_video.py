from itertools import product

import math
import numpy as np
import cv2
import os

import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from mpl_toolkits.mplot3d import Axes3D
from keras.models import model_from_json, Model
from keras import backend as K
from keras.utils import np_utils

from data_helpers import smooth, fit_sin
from train_video import generate_batch, open_video

data_dir = os.getcwd() + '/data/'
video_dir = 'speed_videos/'
annotation_dir = 'speed_annotations/'
kernel_size = 32
kernel_frames = 4
frame_size = 64
window_size = 8

vis_frames = 8
vis_size = 64
vis_iter = 50

use_flow_field = False
grayscale = True


def video_to_flow_field(video):
    flow = np.array([])
    for i in range(len(video) - 1):
        field = get_flow_field(video, i, i + 1)
        flow = np.append(flow, field)
    return np.reshape(flow, (video.shape[0] - 1, video.shape[1], video.shape[2], 2))


def flow_to_rgb(flow):
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3))
    hsv[..., 1] = 255
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(np.uint8(hsv), cv2.COLOR_HSV2RGB)


def get_flow_field(video, i, j):
    prev = cv2.cvtColor(video[i], cv2.COLOR_BGR2GRAY)
    next = cv2.cvtColor(video[j], cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow


def plot_weights(weights):
    print(weights.shape)
    for frame_i in range(kernel_frames):
        fig, axes = plt.subplots(8, 4)
        print('Saving frame', frame_i)
        for i in range(len(weights)):
            r = i // 4
            c = i % 4
            img = weights[i][frame_i]
            bgr = cv2.cvtColor(np.uint8(img), cv2.COLOR_HSV2RGB_FULL)
            axes[r][c].imshow(bgr)
            #axes[r][c].quiver(x, y, u, v, mag, edgecolor='k', width=0.05, pivot='tip')
            axes[r][c].set_xticks([])
            axes[r][c].set_yticks([])
        plt.tight_layout(0.5)
        plt.savefig('saved/' + str(frame_i) + '.png', dpi=250)
        plt.clf()


def plot_clip(clip):
    rows = window_size // 2
    cols = window_size // rows
    fig, axes = plt.subplots(rows, cols)
    for i in range(len(clip)):
        r = i // cols
        c = i % cols
        img = clip[i]
        # img = np.reshape(img, (vis_size, vis_size, 3))
        if use_flow_field:
            img = flow_to_rgb(np.float32(img))
        elif grayscale:
            img = img[..., 0]
            print(img.shape, img.max())
        axes[r][c].imshow(img)
        axes[r][c].set_title(str(i))
        axes[r][c].set_xticks([])
        axes[r][c].set_yticks([])
    plt.tight_layout()
    plt.show()


# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    #x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def plot_conv_layer(model, layer_name, layer_dict, input_video=None):
    visualizations = np.array([])
    layer_output = layer_dict[layer_name].output
    input_img = model.input
    print(layer_output)
    rows = 4
    cols = 4
    num_filters = rows * cols
    active_layers = 0  # keeps track of number of activated layers currently in the visualization
    filter_index = 0

    while active_layers < num_filters and 'video' not in layer_name:
        if input_video is None:
            if use_flow_field:
                noise_batch = np.random.random((1, vis_frames - 1, vis_size, vis_size, 2))
            elif grayscale:
                noise_batch = np.random.normal(1, size=(1, vis_frames, vis_size, vis_size, 1))
            else:
                noise_batch = np.random.normal(1, size=(1, vis_frames, vis_size, vis_size, 3))
        else:
            noise_batch = input_video
        # build a loss function that maximizes the activation
        # of the nth filter of the layer considered
        try:
            loss = K.mean(layer_output[..., filter_index])
        except Exception as e:
            layer_output = layer_dict[layer_name].output
            filter_index = 0
            print(e)
            pass

        # compute the gradient of the input picture wrt this loss
        grads = K.gradients(loss, input_img)[0]

        # normalization trick: we normalize the gradient
        grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

        # this function returns the loss and grads given the input picture
        iterate = K.function([input_img], [loss, grads])

        filter_index += 1

        step = 1.
        # run gradient ascent for 20 steps
        for i in range(vis_iter):
            loss_value, grads_value = iterate([noise_batch])
            if loss_value == 0:
                print('Neuron', filter_index, 'not activated')
                break
            noise_batch += grads_value * step
        if loss_value != 0:
            active_layers += 1
            print(active_layers, '/', num_filters)
            visualizations = np.append(visualizations, noise_batch)

    if use_flow_field:
        print(visualizations.shape)
        visualizations = np.reshape(visualizations, (num_filters, 1, vis_frames - 1, vis_size, vis_size, 2))
    elif grayscale:
        visualizations = np.reshape(visualizations, (num_filters, 1, vis_frames, vis_size, vis_size, 1))
    else:
        visualizations = np.reshape(visualizations, (num_filters, 1, vis_frames, vis_size, vis_size, 3))
    frame_offset = 0
    if use_flow_field:
        frame_offset = -1
    for frame_i in range(vis_frames + frame_offset):
        fig, axes = plt.subplots(rows, cols)
        print('Saving frame', frame_i)
        for i in range(num_filters):
            r = i // cols
            c = i % cols
            frame = visualizations[i][0][frame_i]
            if use_flow_field:
                img = flow_to_rgb(np.float32(frame))
                axes[r][c].imshow(img)
            elif grayscale:
                img = deprocess_image(frame)
                img = np.reshape(img, (vis_size, vis_size))
                axes[r][c].imshow(img, cmap='gray')
            else:
                img = deprocess_image(frame)
                img = np.reshape(img, (vis_size, vis_size, 3))
                axes[r][c].imshow(img)
            axes[r][c].set_title(layer_name + ': ' + str(i))
            axes[r][c].set_xticks([])
            axes[r][c].set_yticks([])
        try:
            os.mkdir('conv_vis/' + layer_name)
        except OSError:
            print('Directory', layer_name, 'already exists')
        plt.tight_layout()
        plt.savefig('conv_vis/' + layer_name + '/' + str(frame_i) + '.png', dpi=300)
        plt.clf()


def predict_test(model):
    video_path = np.random.choice(os.listdir(data_dir + video_dir))
    label_path = data_dir + annotation_dir + video_path.replace('.mp4', '.npy')
    label = np.load(label_path)
    video = open_video(data_dir + video_dir + video_path, window_size=window_size)
    print('Video shape', video.shape)
    print('Video max', video.max())
    num_clips = 0
    smoothed_y = np.zeros(window_size)
    while smoothed_y.all() == 0:
        start_frame = np.random.randint(0, len(video) - window_size)
        if start_frame + window_size < len(video):
            clip = video[start_frame:start_frame + window_size]
            if use_flow_field:
                flow_field = video_to_flow_field(np.uint8(clip))
            label_clip = label[np.where(label < start_frame + window_size)]
            label_clip = label_clip[np.where(label_clip > start_frame)]
            y = np.zeros(window_size)
            for frame in label_clip:
                y[frame - start_frame] = 1
            if y.any() != 0:
                print('fitting:')
                smoothed_y = fit_sin(np.arange(0, window_size), y, window_size)
                print('Smoothed', smoothed_y)
            if use_flow_field:
                pred = model.predict(np.array([flow_field]))
            else:
                pred = model.predict(np.array([clip / 255.]))
            print(pred[0])
            print(y)
            if use_flow_field:
                plot_clip(flow_field)
            else:
                print(clip.max())
                plot_clip(clip / 255.)
            plt.plot(pred[0])
            plt.scatter(np.arange(0, window_size), y)
            plt.plot(smoothed_y)
            plt.show()
            print(np.sum(pred[0]))


# load data
dir = 'models/'
with open(dir + 'model_architecture.json', 'r') as f:
    model = model_from_json(f.read())
# Load weights into the new model
model.load_weights(dir + 'model_weights.h5')
print(model.summary())

predict_test(model)

# get the symbolic outputs of each "key" layer (we gave them unique names).
layer_dict = dict([(layer.name, layer) for layer in model.layers])
print(layer_dict)
for batch in generate_batch(1):
    for layer in layer_dict.keys():
        if 'conv' in layer or 'activation' in layer or 'video' in layer:
        #if 'frames' in layer:
            try:
                #plot_conv_layer(model, layer, layer_dict, input_video=batch[0]['video'])
                plot_conv_layer(model, layer, layer_dict)
            except Exception as e:
                print(e)
                pass
    break

