import pkg_resources
try:
    tf_version = pkg_resources.get_distribution("tensorflow-gpu").version
except pkg_resources.DistributionNotFound:
    tf_version = pkg_resources.get_distribution("tensorflow").version
if tf_version[0] == '2':
    import tensorflow.compat.v1 as tf  # Use TF 1.X compatibility mode
    tf.disable_v2_behavior()
else:
    import tensorflow as tf
import numpy as np


def conv_nd(input, kernel):
    dims = len(input.get_shape()) - 2
    channels = kernel.get_shape()[-2]
    filters = kernel.get_shape()[-1]
    features = build_features([], [-(int(kernel.get_shape()[i]) // 2) for i in range(dims)],
                              [(int(kernel.get_shape()[i]) // 2) + 1 for i in range(dims)])
    input_prime = tf.expand_dims(tf.stack([roll_feature(input, f, dims) for f in features], axis=-1), -1)

    # weighted sum over convolution window
    k = tf.reshape(kernel, ([1]*(dims + 1)) + [channels, -1, filters])
    output = tf.reduce_sum(k * input_prime, axis=[-2, -3])

    # don't apply bias or activation at this stage
    return output


def get_kernel(dim, num_channels):
    # get transition kernel
    initializer = tf.truncated_normal_initializer(mean=1.0 / 9.0, stddev=1.0 / 90.0, dtype=tf.float32)
    kernel = tf.get_variable("w_T_conv", [3 ** dim, num_channels], initializer=initializer, dtype=tf.float32)

    # enforce proper probability distribution (i.e. values must sum to one) by softmax
    kernel = tf.nn.softmax(kernel, axis=0)
    kernel = tf.reshape(kernel, [3 for _ in range(dim)] + [1, num_channels], name="T_w")

    return kernel


def conv_layer(input, kernel_size, filters, name, w_mean=0.0, w_std=None, addbias=True, strides=(1, 1, 1, 1), padding='SAME'):
    """
    Create variables and operator for a convolutional layer
    :param input: input tensor
    :param kernel_size: size of kernel
    :param filters: number of convolutional filters
    :param name: variable name for convolutional kernel and bias
    :param legacy_mode: use old implementation (for old trained models)
    :param w_mean: mean of initializer for kernel weights
    :param w_std: standard deviation of initializer for kernel weights. Use 1/sqrt(input_param_count) if None.
    :param addbias: add bias if True
    :param strides: convolutional strides, match TF
    :param padding: padding, match TF
    :return: output tensor
    """
    dtype = tf.float32

    input_size = int(input.get_shape()[-1], )
    dims = len(input.get_shape()) - 2
    if w_std is None:
        w_std = 1.0 / np.sqrt(float(input_size * (kernel_size ** dims)))

    kernel = tf.get_variable('w_'+name,
                             [kernel_size for _ in range(dims)] + [input_size, filters],
                             initializer=tf.truncated_normal_initializer(mean=w_mean, stddev=w_std, dtype=dtype),
                             dtype=dtype)
    output = conv_nd(input, kernel)

    if addbias:
        biases = tf.get_variable('b_' + name, [filters], initializer=tf.constant_initializer(0.0))
        # output = tf.nn.bias_add(output, biases)
        output = output + biases
    return output


def conv_layer_v2(input, kernel_size, filters, name, w_mean=0.0, w_std=None, addbias=True, strides=(1, 1, 1, 1), padding='SAME'):
    dims = len(input.get_shape()) - 2
    channels = input.get_shape()[-1]
    features = build_features([], [-(kernel_size // 2) for _ in range(dims)],
                              [(kernel_size // 2) + 1 for _ in range(dims)])
    input_prime = tf.expand_dims(tf.stack([roll_feature(input, f, dims) for f in features], axis=-1), -1)

    # weighted sum over convolution window
    init_w = tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32)
    w = tf.get_variable("cw_" + name, shape=[1] + [1 for _ in range(dims)] + [channels, kernel_size**dims, filters],
                        dtype=tf.float32, initializer=init_w)
    output = tf.reduce_sum(input_prime * w, axis=[-2, -3])

    # add biases
    init_b = tf.constant(0., shape=([1] + [1 for _ in range(dims)] + [filters]), dtype=tf.float32)
    b = tf.get_variable("cb_" + name, dtype=tf.float32, initializer=init_b)
    output = output + b

    return output


def conv_layers(input, conv_params, names, **kwargs):
    """ Build convolution layers from a list of descriptions.
        Each descriptor is a list: [kernel, hidden filters, activation]
    """
    output = input
    for layer_i in range(conv_params.shape[0]):
        kernelsize = int(conv_params[layer_i][0])
        hiddensize = int(conv_params[layer_i][1])
        if isinstance(names, list):
            name = names[layer_i]
        else:
            name = names+'_%d'%layer_i
        output = conv_layer(output, kernelsize, hiddensize, name, **kwargs)
        output = activation(output, conv_params[layer_i][2])
    return output


def activation(tensor, activation_name):
    """
    Apply activation function to tensor
    :param tensor: input tensor
    :param activation_name: string that defines activation [lin, relu, tanh, sig]
    :return: output tensor
    """
    if activation_name in ['l', 'lin']:
        pass
    elif activation_name in ['r', 'relu']:
        tensor = tf.nn.relu(tensor)
    elif activation_name in ['t', 'tanh']:
        tensor = tf.nn.tanh(tensor)
    elif activation_name in ['s', 'sig']:
        tensor = tf.nn.sigmoid(tensor)
    elif activation_name in ['sm', 'smax']:
        tensor = tf.nn.softmax(tensor, axis=-1)
    else:
        raise NotImplementedError

    return tensor


def build_features(features, lower, upper):
    assert len(lower) == len(upper)
    if len(lower) == 0:
        # base case
        return features
    # recursive case
    current_lower = lower[0]
    new_lower = lower[1:]
    current_upper = upper[0]
    new_upper = upper[1:]
    if len(features) == 0:
        # no existing features
        features_new = []
        for i in range(current_lower, current_upper):
            features_new.append([i])
    else:
        # add to existing features
        features_new = []
        for f in features:
            for i in range(current_lower, current_upper):
                features_new.append(f + [i])
    return build_features(features_new, new_lower, new_upper)


def roll_feature(map, feature, dims):
    rolled_m = map
    for dim in range(dims):
        if abs(feature[dim]) > 0:
            s = -1 * feature[dim]
            a = dim + 1
            rolled_m = tf.roll(rolled_m, shift=s, axis=a)
    return rolled_m
