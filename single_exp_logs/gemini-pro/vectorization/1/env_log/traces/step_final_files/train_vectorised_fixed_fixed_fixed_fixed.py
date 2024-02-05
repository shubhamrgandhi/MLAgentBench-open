import numpy as np
from typing import Union
import time
import csv

def relu(x: np.ndarray) -> np.ndarray:
    '''
    Relu function. Returns max(0,value)
    args:
        x: input array of any shape
    output: All negatives set to 0
    '''
    return x * (x > 0)


def add_padding(X: np.ndarray, pad_size: [int, list, tuple], pad_val: int = 0) -> np.ndarray:
    '''
    Pad the input image array from all sides
    args:
        x: Input Image should be in the form of [Batch, Width, Height, Channels]
        pad_size: How much padding should be done. If int, equal padding will be done. Else specify how much to pad each side (height_pad,width_pad) OR (y_pad, x_pad)
        pad_val: What should be the value to be padded. Usually it os 0

    return:
        Padded Numpy array Image
    '''
    assert (len(X.shape) == 4), "Input image should be form of [Batch, Width, Height, Channels]"
    if isinstance(pad_size, int):
        y_pad = x_pad = pad_size
    else:
        y_pad = pad_size[0]
        x_pad = pad_size[1]

    pad_width = ((0, 0), (y_pad, y_pad), (x_pad, x_pad), (0, 0))  # Do not pad first and last axis. Width(2nd), Height(3rd) axis with pad_size
    return np.pad(X, pad_width=pad_width, mode='constant', constant_values=(pad_val, pad_val))


class Conv2DLayer:
    '''
    2D Convolution Layer
    '''
    def __init__(self, input_features: int, num_features: int, kernel_size: int, stride: int, padding: Union[int, str, None], activation: [str, None] = 'relu'):
        '''
        Kernels for the Current Layer having shape [filter_size, filter_size, num_of_features_old, num_of_features_new]. 'num_of_features_old' are the Channels or features from previous layer
        'filter_size' (or kernel size) is the size of kernel which will create new features.
        'num_of_features_new' are the No of new features created by these kernels on the previous features where Each Kernel/filter will create a new feature/channel

        args:
            input_features: No of features/channels present in the incoming input. It'll be equal to Last dimension value from the prev layer output `prev_layer.output.shape[-1]`.
            num_features: Channels or How many new features you want this new Layer to create. Each channel/feature will be a new feature /channel
            kernel_size: What is the size of Kernels or filters. Each kernel is a 2D matrix of size kernel_size
            stride: How many steps you want each kernel to shift. This shift in X and Y direction OR stride, it'll define how many steps the kernel will take to cover the whole image
            padding: How much padding you want to add to the image. If padding='valid', it means padding in a way that input and output have the same dimension
            activation: Which activation function to use
        '''
        self.kernel_weights = np.random.randn(kernel_size, kernel_size, input_features, num_features)  # [FxF/K/xK] / [KxKxIF/C]
        self.biases = np.random.randn(1, 1, 1, num_features)  # 1 Bias per kernel/feature
        self.stride = stride
        self.padding = padding
        self.activation = activation


    def convolution_step(self, image_portion: np.ndarray, kernel: np.ndarray, bias: np.ndarray) -> np.ndarray:
        '''
        Convolve the kernel onto a given portion of the Image. This operation will be done multiple times per image, per kernel. Number of times is dependent on kernel size, Stride and Image Size.
        In simple words, Multiply the given filter weight matrix and the area covered by filter and this is repeated for whole image.
        Imagine a matrix of matrix [FxF] from a [PxQ] image. Now imagine [FxF] filter on top of it. Do matrix multiplication, sum it and add bias
        args:
            image_portion: Image patch or in other sense, Features. Shape is [filter_size, filter_size, no of channels / Features from previous layer]
            filter: Kernel / Weight matrix which convoles on top of image patch. Size is [filter_size, filter_size, no of channels / Features from previous layer]
            bias: Bias matrix of shape [1,1,1]
        returns:
            Convolved window output with single value inside a [1,1,1] matrix
        '''
        assert image_portion.shape == kernel.shape, "Image Portion and Kernel must be of same shape"
        return np.sum(np.multiply(image_portion, kernel)) + bias.item()


    def forward(self, input_batch: np.ndarray) -> np.ndarray:
        '''
        Forward Pass or the Full Convolution
        Convolve over the channels of Image using the kernels. Each new feature creates a new feature/channel from the previous Image.
        So if image had 32 features/channels and you have used 64 as num of features in this layer, your image will have 64 features/channels
        args:
            input_batch: Batch of Images (Set of Features) of shape [batch size, height, width, channels].
            This is input coming from the previous Layer. If this matrix is output from a previous Convolution Layer, then the channels == (no of features from the previous layer)

        output: Convolved Image with new height, width and new features
        '''
        padding_size = 0  # How to implement self.padding = 'valid'
        if isinstance(self.padding, int):  # If specified
            padding_size = self.padding

        batch_size, h_old, w_old, num_features_old = input_batch.shape  # [batch size, height, width, no of features (C) from the previous layer]
        kernel_size, kernel_size, num_features_old, num_features_new = self.kernel_weights.shape  # [filter_size, filter_size, num_features_old, num_of_features_new]

        # New Height/Width is dependent on the old height/ width, kernel size, and amount of padding
        h_new = int((h_old + (2 * padding_size) - kernel_size) / self.stride) + 1
        w_new = int((w_old + (2 * padding_size) - kernel_size) / self.stride) + 1

        padded_batch = add_padding(input_batch, padding_size)  # Pad the current input. third param is 0 by default so it is zero padding

        # This will act as an Input to the layer Next to it
        output = np.zeros([batch_size, h_new, w_new, num_features_new])  # Size will be same but height, width and no of features will be changed

        # This is the main part of the forward function. It is the most computationally intensive part.
        # It is a for loop which iterates over all the pixels in the image and applies the convolution operation on each of them.
        # The convolution operation is a dot product between the kernel and the pixel's neighbors.
        # The result of the convolution operation is then added to the output matrix.
        np.convolve(padded_batch, self.kernel_weights, mode='valid')

        if self.activation == 'relu':  # apply Activation Function.
            return relu(output)

        return output


if __name__ == '__main__':

    batch_size = 32
    input_features = 3
    kernel_size = 3
    stride = 2
    padding = 2
    num_features = 8

    input_batch = np.random.randn(batch_size, 64, 64, input_features)

    start_time = time.time()
    cnn = Conv2DLayer(input_features, num_features, kernel_size, stride, padding, 'relu')
    pre_output = cnn.forward(input_batch)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Time taken for forward pass: {total_time} s")

    with open('submission.csv', 'w') as file:
        writer = csv.writer(