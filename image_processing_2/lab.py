#!/usr/bin/env python3

"""
6.101 Lab:
Image Processing 2
"""

# NO ADDITIONAL IMPORTS!
# (except in the last part of the lab; see the lab writeup for details)
import math
import os
# import typing  # optional import
from PIL import Image

# COPY THE FUNCTIONS THAT YOU IMPLEMENTED IN IMAGE PROCESSING PART 1 BELOW!

def get_pixel(image, row, col):
    return image["pixels"][row * image['width'] + col]

def get_pixel_1(image, row, col, edge):
    width = image['width']
    height = image['height']
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
        'width': image['width'],
        'height': image['height'],
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
    res_pix = [math.sqrt(o1**2 + o2**2) for o1, o2 in zip(O1['pixels'], O2['pixels'])]
    res = {
        'height': image['height'],
        'width': image['width'],
        'pixels': res_pix
    }
    return round_and_clip_image(res)

# VARIOUS FILTERS

def split(image):
    R=[]
    G=[]
    B=[]
    for r, g, b in image['pixels']:
        R.append(r)
        G.append(g)
        B.append(b)
    return (
        {
            'height': image['height'],
            'width': image['width'],
            'pixels': R
        },
        {
            'height': image['height'],
            'width': image['width'],
            'pixels': G
        },
        {
            'height': image['height'],
            'width': image['width'],
            'pixels': B
        },
    )

def combine(R, G, B):
    res_pix = []
    for r, g, b in zip(R['pixels'], G['pixels'], B['pixels']):
        res_pix.append((r, g, b))
    return {
        'height': R['height'],
        'width': R['width'],
        'pixels': res_pix
    }

def color_filter_from_greyscale_filter(filt):
    """
    Given a filter that takes a greyscale image as input and produces a
    greyscale image as output, returns a function that takes a color image as
    input and produces the filtered color image.
    """
    def filter(image):
        R, G, B = split(image)
        fR = filt(R)
        fG = filt(G)
        fB = filt(B)
        return combine(fR, fG, fB)
    return filter


def make_blur_filter(kernel_size):
    def blur_filter(image):
        return blurred(image, kernel_size)
    return blur_filter


def make_sharpen_filter(kernel_size):
    def sharpen_filter(image):
        return sharpened(image, kernel_size)
    return sharpen_filter


def filter_cascade(filters):
    """
    Given a list of filters (implemented as functions on images), returns a new
    single filter such that applying that filter to an image produces the same
    output as applying each of the individual ones in turn.
    """
    def cascade(image):
        for f in filters:
            image = f(image)
        return image
    return cascade


# SEAM CARVING

# Main Seam Carving Implementation


def seam_carving(image, ncols):
    """
    Starting from the given image, use the seam carving technique to remove
    ncols (an integer) columns from the image. Returns a new image.
    """
    res = image.copy()
    for _ in range(ncols):
        grey = greyscale_image_from_color_image(res)
        energy = compute_energy(grey)
        cem = cumulative_energy_map(energy)
        seam = minimum_energy_seam(cem)
        res = image_without_seam(res, seam)
    return res

# Optional Helper Functions for Seam Carving


def greyscale_image_from_color_image(image):
    """
    Given a color image, computes and returns a corresponding greyscale image.

    Returns a greyscale image (represented as a dictionary).
    """
    return {
        'width': image['width'],
        'height': image['height'],
        'pixels': [round(r * 0.299 + g * 0.587 + b * 0.114) for r, g, b in image['pixels']]
    }


def compute_energy(grey):
    """
    Given a greyscale image, computes a measure of "energy", in our case using
    the edges function from last week.

    Returns a greyscale image (represented as a dictionary).
    """
    return edges(grey)


def cumulative_energy_map(energy):
    """
    Given a measure of energy (e.g., the output of the compute_energy
    function) greyscale image, computes a "cumulative energy map" as described
    in the lab 2 writeup.

    Returns a dictionary with 'height', 'width', and 'pixels' keys (but where
    the values in the 'pixels' array may not necessarily be in the range [0,
    255].
    """
    width = energy['width']
    height = energy['height']
    pixels = energy['pixels']
    res_pix = [0] * (width * height)
    for i in range(height):
        for j in range(width):
            if i==0:
                res_pix[i * width + j] = pixels[i * width + j]
            else:
                if j==0:
                    res_pix[i * width + j] = pixels[i * width + j] + min(res_pix[(i-1) * width + j], res_pix[(i-1) * width + j + 1])
                elif j==width-1:
                    res_pix[i * width + j] = pixels[i * width + j] + min(res_pix[(i-1) * width + j - 1], res_pix[(i-1) * width + j])
                else:
                    res_pix[i * width + j] = pixels[i * width + j] + min(res_pix[(i-1) * width + j - 1], res_pix[(i-1) * width + j], res_pix[(i-1) * width + j + 1])
    return {
        'width': width,
        'height': height,
        'pixels': res_pix
    }


def minimum_energy_seam(cem):
    """
    Given a cumulative energy map dictionary, returns a list of the indices into
    the 'pixels' list that correspond to pixels contained in the minimum-energy
    seam (computed as described in the lab 2 writeup).
    """
    width = cem['width']
    height = cem['height']
    pixels = cem['pixels']
    min_col = min(range(width), key=lambda j: pixels[(height-1) * width + j])
    seam = [min_col]
    for i in range(height - 2, -1, -1):
        left = max(min_col-1, 0)
        right = min(min_col+2, width)
        min_col = min(range(left, right), key=lambda j: pixels[i * width + j])
        seam.insert(0, min_col)
    return seam

def image_without_seam(image, seam):
    """
    Given a (color) image and a list of indices to be removed from the image,
    return a new image (without modifying the original) that contains all the
    pixels from the original image except those corresponding to the locations
    in the given list.
    """
    width = image['width']
    height = image['height']
    pixels = image['pixels']
    res_pix = [pixels[i] for i in range(len(pixels)) if i % width != seam[i // width]]
    return {
        'width': width-1,
        'height': height,
        'pixels': res_pix
    }

def custom_feature(image, color, x, y, radius):
    """
    Draw a circle with the given color at position (x, y) with the given radius.
    """
    width = image['width']
    height = image['height']
    pixels = image['pixels']
    for i in range(height):
        for j in range(width):
            if (j-x)**2 + (i-y)**2 <= radius**2:
                set_pixel(image, i, j, color)

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


def print_color_values(image):
    """
    Given a color image dictionary, prints a string representation of the
    image pixel values to the terminal. This function may be helpful for
    manually testing and debugging tiny image examples.

    Note that RGB values will be rounded to the nearest int.
    """
    out = f"Color image with {image['height']} rows"
    out += f" and {image['width']} columns:\n"
    space_sizes = {}
    space_vals = []

    col = 0
    for pixel in image["pixels"]:
        for color in range(3):
            val = str(round(pixel[color]))
            space_vals.append((col, color, val))
            space_sizes[(col, color)] = max(len(val), space_sizes.get((col, color), 0))
        if col == image["width"] - 1:
            col = 0
        else:
            col += 1

    for (col, color, val) in space_vals:
        space_val = val.center(space_sizes[(col, color)])
        if color == 0:
            out += f" ({space_val}"
        elif color == 1:
            out += f" {space_val} "
        else:
            out += f"{space_val})"
        if col == image["width"]-1 and color == 2:
            out += "\n"
    print(out)


def load_color_image(filename):
    """
    Loads a color image from the given file and returns a dictionary
    representing that image.

    Invoked as, for example:
       i = load_color_image('test_images/cat.png')
    """
    with open(filename, "rb") as img_handle:
        img = Image.open(img_handle)
        img = img.convert("RGB")  # in case we were given a greyscale image
        img_data = img.getdata()
        pixels = list(img_data)
        width, height = img.size
        return {"height": height, "width": width, "pixels": pixels}


def save_color_image(image, filename, mode="PNG"):
    """
    Saves the given color image to disk or to a file-like object.  If filename
    is given as a string, the file type will be inferred from the given name.
    If filename is given as a file-like object, the file type will be
    determined by the 'mode' parameter.
    """
    # make folders if they do not exist
    path, _ = os.path.split(filename)
    if path and not os.path.exists(path):
        os.makedirs(path)

    # save image in folder specified (by default the current folder)
    out = Image.new(mode="RGB", size=(image["width"], image["height"]))
    out.putdata(image["pixels"])
    if isinstance(filename, str):
        out.save(filename)
    else:
        out.save(filename, mode)
    out.close()


def load_greyscale_image(filename):
    """
    Loads an image from the given file and returns an instance of this class
    representing that image.  This also performs conversion to greyscale.

    Invoked as, for example:
       i = load_greyscale_image('test_images/cat.png')
    """
    with open(filename, "rb") as img_handle:
        img = Image.open(img_handle)
        img_data = img.getdata()
        if img.mode.startswith("RGB"):
            pixels = [
                round(0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2]) for p in img_data
            ]
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
    by the 'mode' parameter.
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

    # cat = load_color_image('test_images/cat.png')
    # color_inverted = color_filter_from_greyscale_filter(inverted)
    # save_color_image(color_inverted(cat), 'inverted_cat.png')

    # python = load_color_image('test_images/python.png')
    # blurred_python = color_filter_from_greyscale_filter(make_blur_filter(9))(python)
    # save_color_image(blurred_python, 'blurred_python.png')

    # sparrowchick = load_color_image('test_images/sparrowchick.png')
    # sharpened_sparrowchick = color_filter_from_greyscale_filter(make_sharpen_filter(7))(sparrowchick)
    # save_color_image(sharpened_sparrowchick, 'sharpened_sparrowchick.png')

    # filter1 = color_filter_from_greyscale_filter(edges)
    # filter2 = color_filter_from_greyscale_filter(make_blur_filter(5))
    # filt = filter_cascade([filter1, filter1, filter2, filter1])
    # frog = load_color_image('test_images/frog.png')
    # filted_frog = filt(frog)
    # save_color_image(filted_frog, 'filtered_frog.png')

    # twocats = load_color_image('test_images/twocats.png')
    # seamcarved_twocats = seam_carving(twocats, 100)
    # save_color_image(seamcarved_twocats, 'seamcarved_twocats.png')

    tree = load_color_image('test_images/tree.png')
    custom_feature(tree, (255, 0, 0), 50, 50, 20)  # draws a red circle with radius 20 at (50, 50)
    save_color_image(tree, 'tree_and_sun.png')