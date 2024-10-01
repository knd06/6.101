#!/usr/bin/env python3

"""
6.101 Lab:
Image Processing
"""

import math
import os
from PIL import Image

# NO ADDITIONAL IMPORTS ALLOWED!


def get_pixel(image, row, col):
    return image["pixels"][row * image['width'] + col]

def get_pixel_1(image, row, col, edge):
    height = image['height']
    width = image['width']
    if edge=='zero':
        if row < 0 or row >= height or col < 0 or col >= width: 
            return 0
        return get_pixel(image, row, col)
    if edge=='extend':
        row = max(0, min(row, height-1))
        col = max(0, min(col, width-1))
    if edge=='wrap':
        row = row % height
        col = col % width
    return get_pixel(image, row, col)


def set_pixel(image, row, col, color):
    image["pixels"][row * image['width'] + col] = color


def apply_per_pixel(image, func):
    result = {
        "height": image["height"],
        "width": image["width"],
        "pixels": [0] * (image['height'] * image['width']),
    }
    for row in range(image["height"]):
        for col in range(image["width"]):
            color = get_pixel(image, row, col)
            new_color = func(color)
            set_pixel(result, row, col, new_color)
    return result


def inverted(image):
    return apply_per_pixel(image, lambda color: 255-color)


# HELPER FUNCTIONS

def correlate(image, kernel, boundary_behavior):
    """
    Compute the result of correlating the given image with the given kernel.
    `boundary_behavior` will one of the strings "zero", "extend", or "wrap",
    and this function will treat out-of-bounds pixels as having the value zero,
    the value of the nearest edge, or the value wrapped around the other edge
    of the image, respectively.

    if boundary_behavior is not one of "zero", "extend", or "wrap", return
    None.

    Otherwise, the output of this function should have the same form as a 6.101
    image (a dictionary with "height", "width", and "pixels" keys), but its
    pixel values do not necessarily need to be in the range [0,255], nor do
    they need to be integers (they should not be clipped or rounded at all).

    This process should not mutate the input image; rather, it should create a
    separate structure to represent the output.

    DESCRIBE YOUR KERNEL REPRESENTATION HERE
    """
    if not(boundary_behavior=='zero' or boundary_behavior =='extend' or boundary_behavior=='wrap'):
        return None
    res_pix = [0] * (image['width'] * image['height'])
    kernel_sz = len(kernel)
    half_sz = int((kernel_sz-1)/2)
    
    for r in range(image['height']):
        for c in range(image['width']):
            for kr in range(-half_sz, half_sz+1):
                for kc in range(-half_sz, half_sz+1):
                    res_pix[r * image['width'] + c] += (get_pixel_1(image, r + kr, c + kc, boundary_behavior) * kernel[kr + half_sz][kc + half_sz])
    return {
        'height': image['height'],
        'width': image['width'],
        'pixels': res_pix
    }

def round_and_clip_image(image):
    """
    Given a dictionary, ensure that the values in the "pixels" list are all
    integers in the range [0, 255].

    All values should be converted to integers using Python's `round` function.

    Any locations with values higher than 255 in the input should have value
    255 in the output; and any locations with values lower than 0 in the input
    should have value 0 in the output.
    """
    return {
        'height':image['height'],
        'width': image['width'],
        'pixels': [max(0, min(255, round(x))) for x in image['pixels']]
    }


# FILTERS

def blurred(image, kernel_size):
    """
    Return a new image representing the result of applying a box blur (with the
    given kernel size) to the given input image.

    This process should not mutate the input image; rather, it should create a
    separate structure to represent the output.
    """
    # first, create a representation for the appropriate n-by-n kernel (you may
    # wish to define another helper function for this)

    # then compute the correlation of the input image with that kernel

    # and, finally, make sure that the output is a valid image (using the
    # helper function from above) before returning it.
    blur_kernel = [[1.0 / (kernel_size * kernel_size)] * kernel_size for _ in range(kernel_size)]
    res = correlate(image, blur_kernel, boundary_behavior='extend')
    return round_and_clip_image(res)

def sharpened(image, n):
    """
    Return a new image representing the result of applying a sharpening filter to the given input image.

    This process should not mutate the input image; rather, it should create a
    separate structure to represent the output.
    """
    blurred_im = blurred(image, n)
    res_pix = [min(max(2 * org - blur, 0), 255) for org, blur in zip(image['pixels'], blurred_im['pixels'])]
    return {
        'height': image['height'],
        'width': image['width'],
        'pixels': res_pix
    }

def edges(image):
    """
    Return a new image representing the result of applying Sobel operator to the given input image.

    This function computes the gradient of the image intensity at each pixel
    using two kernels: detecting horizontal / vertical edges. The magnitude of the gradient is calculated to produce 
    an output image that highlights the edges present in the input image.

    This process should not mutate the input image; rather, it should create a
    separate structure to represent the output.
    """
    K1 = [[-1, -2, -1],
          [0, 0, 0],
          [1, 2, 1]]
    K2 = [[-1, 0, 1],
          [-2, 0, 2],
          [-1, 0, 1]]
    O1 = correlate(image, K1, 'extend')
    O2 = correlate(image, K2, 'extend')
    res_pix = []
    for o1, o2 in zip(O1['pixels'], O2['pixels']):
        res_pix.append(math.sqrt(o1**2 + o2**2))
    res = {
        'height': image['height'],
        'width': image['width'],
        'pixels': res_pix
    }
    return round_and_clip_image(res)

# HELPER FUNCTIONS FOR DISPLAYING, LOADING, AND SAVING IMAGES

def print_greyscale_values(image):
    """
    Given a greyscale image dictionary, prints a string representation of the
    image pixel values to the terminal. This function may be helpful for
    manually testing and debugging tiny image examples.

    Note that pixel values that are floats will be rounded to the nearest int.
    """
    out = f"Greyscale image with {image['height']} rows"
    out += f" and {image['width']} columns:\n "
    space_sizes = {}
    space_vals = []

    col = 0
    for pixel in image["pixels"]:
        val = str(round(pixel))
        space_vals.append((col, val))
        space_sizes[col] = max(len(val), space_sizes.get(col, 2))
        if col == image["width"] - 1:
            col = 0
        else:
            col += 1

    for (col, val) in space_vals:
        out += f"{val.center(space_sizes[col])} "
        if col == image["width"]-1:
            out += "\n "
    print(out)


def load_greyscale_image(filename):
    """
    Loads an image from the given file and returns a dictionary
    representing that image.  This also performs conversion to greyscale.

    Invoked as, for example:
       i = load_greyscale_image("test_images/cat.png")
    """
    with open(filename, "rb") as img_handle:
        img = Image.open(img_handle)
        img_data = img.getdata()
        if img.mode.startswith("RGB"):
            pixels = [round(.299 * p[0] + .587 * p[1] + .114 * p[2])
                      for p in img_data]
        elif img.mode == "LA":
            pixels = [p[0] for p in img_data]
        elif img.mode == "L":
            pixels = list(img_data)
        else:
            raise ValueError(f"Unsupported image mode: {img.mode}")
        width, height = img.size
        return {"height": height, "width": width, "pixels": pixels}


def save_greyscale_image(image, filename, mode="PNG"):
    """
    Saves the given image to disk or to a file-like object.  If filename is
    given as a string, the file type will be inferred from the given name.  If
    filename is given as a file-like object, the file type will be determined
    by the "mode" parameter.
    """
    # make folders if they do not exist
    path, _ = os.path.split(filename)
    if path and not os.path.exists(path):
        os.makedirs(path)

    # save image in folder specified (by default the current folder)
    out = Image.new(mode="L", size=(image["width"], image["height"]))
    out.putdata(image["pixels"])
    if isinstance(filename, str):
        out.save(filename)
    else:
        out.save(filename, mode)
    out.close()


if __name__ == "__main__":
    # code in this block will only be run when you explicitly run your script,
    # and not when the tests are being run.  this is a good place for
    # generating images, etc.
    
    # bluegill = load_greyscale_image('test_images/bluegill.png')
    # save_greyscale_image(inverted(bluegill), 'inverted_bluegill.png')

    # pigbird = load_greyscale_image('test_images/pigbird.png')
    # kernel = [[1 if (r == 2 and c == 0) else 0 for c in range(13)] for r in range(13)]
    # for b in ['zero', 'extend', 'wrap']:
    #     output_im = correlate(pigbird, kernel, boundary_behavior=b)
    #     rounded_im = round_and_clip_image(output_im)
    #     save_greyscale_image(rounded_im, f'pigbird_{b}.png')

    # cat = load_greyscale_image('test_images/cat.png')
    # save_greyscale_image(blurred(cat, 13), 'blurred_cat.png')

    # python = load_greyscale_image("test_images/python.png")
    # save_greyscale_image(sharpened(python, 11), "sharpened_python.png")

    construct = load_greyscale_image("test_images/construct.png")
    save_greyscale_image(edges(construct), "edges_of_construct.png")