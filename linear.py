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


def linear_layer(input, output_size, name, w_mean=0.0, w_std=None):
    """
    Create variables and operator for a linear layer
    :param input: input tensor
    :param output_size: output size, number of hidden units
    :param name: variable name for linear weights and bias
    :param w_mean: mean of initializer for kernel weights
    :param w_std: standard deviation of initializer for kernel weights. Use 1/sqrt(input_param_count) if None.
    :return: output tensor
    """
    dtype = tf.float32

    if w_std is None:
        w_std = 1.0 / np.sqrt(float(np.prod(input.get_shape().as_list()[1])))

    w = tf.get_variable('w_' + name,
                        [input.get_shape()[1], output_size],
                        initializer=tf.truncated_normal_initializer(mean=w_mean, stddev=w_std, dtype=dtype),
                        dtype=dtype)

    b = tf.get_variable("b_" + name, [output_size], initializer=tf.constant_initializer(0.0))

    output = tf.matmul(input, w) + b

    return output


def fc_layers(input, conv_params, names, **kwargs):
    """ Build convolution layers from a list of descriptions.
        Each descriptor is a list: [size, _, activation]
    """
    output = input
    for layer_i in range(conv_params.shape[0]):
        size = int(conv_params[layer_i][0])
        if isinstance(names, list):
            name = names[layer_i]
        else:
            name = names+'_%d'%layer_i
        output = linear_layer(output, size, name, **kwargs)
        output = activation(output, conv_params[layer_i][-1])
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