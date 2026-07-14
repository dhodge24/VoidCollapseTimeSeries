
import cupy as cp
from cupyx.scipy.ndimage import zoom
# from skimage.transform import resize

from utils.utilities import cropToCenter
from utils.utilities import padToSize

from parameters.DataDimensions import DataDimensions
from parameters.Padding import Padding
from preprocessing.Boundaries import Boundaries

from constraints.window_2d import get_2d_window


def flip_and_pad(image, data_dimensions, padding_options):
    orig_size = data_dimensions.fov_size

    # Crop the image to center
    fov = cropToCenter(image, orig_size)

    # Handle different padding modes
    if padding_options.padding_mode.value == Padding.PaddingMode.MIRROR_ALL.value:
        # Mirror padding on all sides
        padding = ((orig_size[1]-1, orig_size[1]-1), (orig_size[0]-1, orig_size[0]-1))
        mirrored_image = cp.pad(fov, padding, mode='reflect')
    elif padding_options.padding_mode.value == Padding.PaddingMode.MIRROR_LEFT.value:
        # Mirror padding only on the left
        padding = ((0, 0), (orig_size[0] - 1, 0))
        mirrored_image = cp.pad(fov, padding, mode='reflect')

    # Handle rolling the image for MIRROR_LEFT mode
    if padding_options.padding_mode.value == Padding.PaddingMode.MIRROR_LEFT.value:
        mirrored_image = cp.roll(mirrored_image, shift=-int(orig_size[0] / 2), axis=1)

    # Pad to the final size with a constant value
    image = padToSize(
        img=mirrored_image,
        outputSize=image.shape,
        padMethod='constant',
        padType='both',
        padValue=padding_options.i0
    )

    return image

def process_image(image, padding_options:Padding, data_dimensions:DataDimensions):
    image_padding = padding_options.i0

    if padding_options.padding_factor < 4:
        padded_size = tuple([round(x * 4) for x in data_dimensions.fov_size])
    else:
        padded_size = data_dimensions.total_size
    padded_size_extern = data_dimensions.total_size

    original_size = image.shape
    image = image[padding_options.cutting_band:original_size[0] - padding_options.cutting_band,
            padding_options.cutting_band:original_size[1] - padding_options.cutting_band]
    # Filter
    if padding_options.down_sampling_factor > 1:
        down_sampled_size = tuple([round(x / padding_options.down_sampling_factor) for x in image.shape])
        # Calculate zoom factors
        zoom_factors = [down_size / orig_size for down_size, orig_size in zip(down_sampled_size, image.shape)]
        # Resize using zoom
        image = zoom(image, zoom_factors, mode='reflect')
        # Resize using skimage
        # image = resize(image, output_shape=down_sampled_size, mode='reflect', anti_aliasing=True)

    cropped_size = image.shape

    # Pad image
    image = padToSize(
        img=image,
        outputSize=padded_size,
        padMethod='constant',
        padType='both',
        padValue=image_padding
    )

    i0 = image_padding * cp.ones(image.shape, dtype=cp.float32)

    if padding_options.padding_mode.value == Padding.PaddingMode.MIRROR_ALL.value:
        window_width_x = (int(cropped_size[0] / 2) + int(cropped_size[0] / 2) % 2,
                         int(cropped_size[0] / 2) + int(cropped_size[0] / 2) % 2)
        window_width_y = (int(cropped_size[1] / 2) + int(cropped_size[1] / 2) % 2,
                         int(cropped_size[1] / 2) + int(cropped_size[1] / 2) % 2)

        image = flip_and_pad(image, data_dimensions, padding_options)

        boundaries = Boundaries(padded_size, cropped_size)

        window = get_2d_window(image.shape,
                            [(boundaries.start_top_x, boundaries.end_bottom_x),
                             (boundaries.start_left_y,boundaries.end_right_y)],
                            [window_width_x, window_width_y], data_dimensions.window_type)

    elif padding_options.padding_mode.value == Padding.PaddingMode.MIRROR_LEFT.value:
        window_width_x = (int(cropped_size[0] / 2) + int(cropped_size[0] / 2) % 2, 40)
        window_width_y = (80, 80)

        image = flip_and_pad(image, data_dimensions, padding_options)

        boundaries = Boundaries(padded_size, cropped_size)

        window = get_2d_window(image.shape,
                            [(boundaries.start_middle_x, boundaries.end_middle_x),
                             (boundaries.start_left_y, boundaries.end_middle_y)],
                            [window_width_x, window_width_y], data_dimensions.window_type)

    elif padding_options.padding_mode.value == Padding.PaddingMode.NORMAL.value:
        window_width_x = (80, 80)
        window_width_y = (80, 80)

        boundaries = Boundaries(padded_size, cropped_size)

        window = get_2d_window(image.shape,
                            [(boundaries.start_middle_x, boundaries.end_middle_x),
                             (boundaries.start_middle_y, boundaries.end_middle_y)],
                            (window_width_x, window_width_y), data_dimensions.window_type)

    else:
        raise RuntimeError("Padding mode not implemented")

    data_dimensions.window = window

    image = (image - i0) * window + i0

    if padded_size_extern < padded_size:
        image = cropToCenter(image, padded_size_extern)
        data_dimensions.window = cropToCenter(data_dimensions.window, padded_size_extern)

    return image
