
from constraints.window_functions import *


def check_edge_conditions(start_1, end_2, window_steepness, data_length):
    if start_1 >= 0:
        window_start_1 = 0
    else:
        window_start_1 = -start_1
        start_1 = 0

    if end_2 < data_length:
        window_end_2 = (int(window_steepness) - 1)
    else:
        window_end_2 = (int(window_steepness) - (end_2 - data_length) - 1)
        end_2 = data_length - 1

    return start_1, end_2, window_start_1, window_end_2


def get_2d_window_from_function(data_shape, support, window_steepness, window_func):
    mask = cp.zeros(data_shape)

    window_indices_x_1 = cp.linspace(0, window_steepness[0][0], num=window_steepness[0][0])
    window_values_x_1 = window_func(window_indices_x_1, window_steepness[0][0])

    window_indices_x_2 = cp.linspace(0, window_steepness[0][1], num=window_steepness[0][1])
    window_values_x_2 = window_func(window_indices_x_2, window_steepness[0][1])

    window_indices_y_1 = cp.linspace(0, window_steepness[1][0], num=window_steepness[1][0])
    window_values_y_1 = window_func(window_indices_y_1, window_steepness[1][0])

    window_indices_y_2 = cp.linspace(0, window_steepness[1][1], num=window_steepness[1][1])
    window_values_y_2 = window_func(window_indices_y_2, window_steepness[1][1])

    start_x_1 = support[0][0]
    end_x_1 = support[0][0] + int(window_steepness[0][0] / 2)

    start_x_2 = support[0][1] - int(window_steepness[0][1] / 2)
    end_x_2 = support[0][1]

    start_y_1 = support[1][0]
    end_y_1 = support[1][0] + int(window_steepness[1][0] / 2)

    start_y_2 = support[1][1] - int(window_steepness[1][1] / 2)
    end_y_2 = support[1][1]

    start_x_1, end_x_2, window_start_x_1, window_end_x_2 = check_edge_conditions(start_x_1, end_x_2,
                                                                                 window_steepness[0][1], data_shape[0])
    start_y_1, end_y_2, window_start_y_1, window_end_y_2 = check_edge_conditions(start_y_1, end_y_2,
                                                                                 window_steepness[1][1], data_shape[1])

    for i in range(start_y_1, end_y_2):
        mask[start_x_1:end_x_1, i] = window_values_x_1[window_start_x_1:int(window_steepness[0][0] / 2)]
        mask[start_x_2:end_x_2, i] = window_values_x_2[int(window_steepness[0][1] / 2) - 1:window_end_x_2]
        mask[end_x_1:start_x_2, i] = 1

    for i in range(start_x_1, end_x_2):
        mask[i, start_y_1:end_y_1] *= window_values_y_1[window_start_y_1:int(window_steepness[1][0] / 2)]
        mask[i, start_y_2:end_y_2] *= window_values_y_2[int(window_steepness[1][1] / 2) - 1:window_end_y_2]

    return mask


def get_2d_window(data_shape, support, window_steepness, window_func_name):
    if window_func_name == "hanning":
        return get_2d_window_from_function(data_shape, support, window_steepness, hanning)
    elif window_func_name == "hamming":
        return get_2d_window_from_function(data_shape, support, window_steepness, hamming)
    elif window_func_name == "blackman":
        return get_2d_window_from_function(data_shape, support, window_steepness, blackman)
    else:
        mask = cp.zeros(data_shape)
        mask[slice(support[0][0], support[0][1]), slice(support[1][0], support[1][1])] = 1
        return mask
