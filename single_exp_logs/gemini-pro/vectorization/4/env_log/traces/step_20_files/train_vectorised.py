import numpy as np
from typing import Union
import time
import csv

def relu(x:np.ndarray)->np.ndarray:
    '''
    Relu activation function. Returns max(0,value)
    args:
        x: input array of any shape
    output: All negatives clipped to 0 
    '''
    return np.where(x > 0, x, 0)


def add_padding(X:np.ndarray, pad_size:Union[int,list,tuple], pad_val:int=0)->np.ndarray:
    '''
    Pad the input image array equally from all sides
    args:
        x: Input Image should be in the form of [Batch, Width, Height, Channels]
        pad_size: How much padding should be done. If int, equal padding will done. Else specify how much to pad each side (height_pad,width_pad) OR (y_pad, x_pad)
        pad_val: What should be the value to be padded. Usually it os 0 padding
    return:
        Padded Numpy array Image
    '''
    assert (len(X.shape) == 4), "Input image should be form of [Batch, Width, Height, Channels]"
    pad_width = ((0,0), (pad_size,pad_size), (pad_size,pad_size), (0,0)) # Do not pad first and last axis. Pad Width(2nd), Height(3rd) axis with  pad_size
    return np.pad(X, pad_width = pad_width, mode = 'constant', constant_values = (pad_val,pad_val))


class Conv2DLayer:
    '''
    2D Convolution Layer
    '''
    def __init__(self,input_channels:int, num_filters:int, kernel_size:int, stride:int, padding:Union[str,None], activation:Union[None,str]='relu'):
        '''
        Kernal Matrix for the Current Layer having shape [filter_size, filter_size, num_of_features_old, num_of_filters_new]. 'num_of_features_old' are the Channels or features from previous layer
        'filter_size' (or kernel size) is the size of filters which will detect new features. 
        'num_of_filters_new' are the No of new features detected by these kernels on the previous features where Each Kernel/filter will detect a new Feature /channel

        args:
            input_channels: No of features/channels present in the incoming input. It'll be equal to Last dimension value from the prev layer output `previous_layer.output.shape[-1]`
            num_filters: Output Channels or How many new features you want this new Layer to Detect. Each Filter/kernel will detect a new Feature /channel
            kernel_size: What is the size of Kernels or Filters. Each Filter a 2D Square Matrix of size kernel_size
            stride: How many pixels you want each kernel to shift. Same shift in X and Y direction OR indirectly, it'll define how many iterations the kernel will take to convolve over the whole image
            padding: How much padding you want to add to the image. If padding='same', it means padding in a way that input and output have the same dimension
            activation: Which activation to use
        '''
        self.kernel_matrices = np.random.randn(kernel_size, kernel_size, input_channels, num_filters) # Complete Weight/Kernel Matrix
        self.biases = np.random.randn(1, 1, 1, num_filters) # 1 Bias per Channel/feature/filter
        self.stride = stride
        self.padding = padding
        self.activation = activation


    def convolution_step(self,image_portion:np.ndarray,kernel_matrix:np.ndarray,bias:np.ndarray)->np.ndarray:
        '''
        Convolve the Filter onto a given portion of the Image. This operation will be done multiple times per image, per kernel. Number of times is dependent on Window size, Stride and Image Size.
        In simple words, Multiply the given filter weight matrix and the area covered by filter and this is repeated for whole image.
        Imagine a slice of matrix  [FxF] from a [PxQ] shaped image. Now imagine [Fxf] filter on top of it. Do matrix multiplication, summation and add bias
        args:
            image_portion: Image Matrix or in other sense, Features. Shape is [filter_size, filter_size, no of channels / Features from previous layer]
            filter: Filter / Kernel weight Matrix which convolves on top of image slice. Size is [filter_size, filter_size, no of channels / Features from previous layer]
            bias: Bias matrix of shape [1,1,1]
        returns: 
            Convolved window output with single floating value inside a [1,1,1] matrix
        '''
        assert image_portion.shape == kernel_matrix.shape , "Image Portion and Filter must be of same shape"
        return np.sum(np.tensordot(image_portion,kernel_matrix,axes=([1,2],[1,2]))) + bias.astype('float')


    def forward(self,features_batch:np.ndarray)->np.ndarray:
        '''
        Forward Pass or the Full Convolution
        Convolve over the batch of Image using the filters. Each new Filter produces a new Feature/channel from the previous Image. 
        So if image had 32 features/channels and you have used 64 as num of filters in this layer, your image will have 64 features/channels
        args:
            features_batch: Batch of Images (Batch of Features) of shape [batch size, height, width, channels]. 
            This is input coming from the previous Layer. If this matrix is output from a previous Convolution Layer, then the channels == (no of features from the previous layer)

        output: Convolved Image batch with new height, width and new detected features
        '''
        padding_size = 0 # How to implement self.padding = 'same'?
        if isinstance(self.padding, int): # If specified padding
            padding_size = self.padding
        
        batch_size, h_old, w_old, num_features_old = features_batch.shape # [batch size, height, width, no of features (channels) from the previous layer]
        filter_size, filter_size, num_features_old, num_of_filters_new = self.kernel_matrices.shape # [filter_size, filter_size, num_features_old, num_of_filters_new]

        # New Height/Width is dependent on the old height/ width, stride, filter size, and amount of padding
        h_new = int((h_old + (2 * padding_size) - filter_size) / self.stride) + 1
        w_new = int((w_old + (2 * padding_size) - filter_size) / self.stride) + 1

        padded_batch = add_padding(features_batch, padding_size) # Pad the current input. third param is 0 by default so it is zero padding

        # This will act as an Input to the layer Next to it
        output = np.zeros([batch_size, h_new, w_new, num_of_filters_new]) # batch size will be same but height, width and no of filters will be changed

        # Vectorised version of the forward pass
        output = np.einsum('bijk,ijkl->bijl', padded_batch, self.kernel_matrices) + self.biases

        if self.activation == 'relu':
            output = relu(output)

        return output