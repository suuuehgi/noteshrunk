#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
VERSION = '1.5.0'

import argparse
from concurrent.futures import ThreadPoolExecutor
import logging
import os
from pathlib import Path
import random
import re
import shutil
import string
import sys
import subprocess
import tempfile

import argcomplete
import numpy as np
from PIL import Image
from scipy.ndimage import median_filter
from skimage import io
from sklearn.cluster import KMeans
from skimage.filters import unsharp_mask
from skimage.morphology import binary_opening, square, disk


def parse_args():
    """
    Parses command line arguments.

    Returns:
        argparse.Namespace: A namespace that holds the arguments as attributes.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Compress scanned documents')

    parser.add_argument(
        'files',
        nargs='+',
        help='Input image file paths.')
    parser.add_argument(
        '-o',
        '--output',
        default='output.pdf',
        type=Path,
        help='Output PDF file path.')
    parser.add_argument(
        '-w',
        '--white_background',
        action='store_true',
        default=False,
        help='Use white background instead of dominant color.')
    parser.add_argument(
        '-s',
        '--saturate',
        action='store_true',
        default=False,
        help='Maximize saturation in the output image.')
    parser.add_argument(
        '-n',
        '--n_colors',
        type=int,
        default=8,
        help='Number of colors in the palette.')
    parser.add_argument(
        '-d',
        '--dpi',
        type=int,
        default=300,
        help='DPI value of the input image/-s')
    parser.add_argument(
        "-q",
        "--quality",
        type=int,
        default=75,
        choices=range(1, 101),  # Allow values between 1 and 100 (inclusive)
        metavar="[1-100]",
        help='JPEG quality of the images embedded in the PDF')
    parser.add_argument(
        '-l',
        '--local_palette',
        action='store_true',
        default=False,
        help='Create an individual color palette for each image (by sampling a -p percentage of the pixels of that image) instead of a global palette (by sampling a -p percentage of the pixels of each input image).')
    parser.add_argument(
        "-p",
        "--percentage",
        type=float,
        default=100,
        help="Percentage of pixels to sample from each image.")
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=os.cpu_count(),
        help="Number of processes to use (default: number of CPU cores)")
    parser.add_argument(
        '-y',
        '--overwrite',
        action='store_true',
        default=False,
        help='Overwrite existing files without asking.')
    parser.add_argument(
        "-ts",
        "--threshold_saturation",
        type=int,
        default=15,
        choices=range(0, 101),  # Allow values between 0 and 100
        metavar="[1-100]",
        help="HSV saturation threshold (in percent) used for the background detection.")
    parser.add_argument(
        "-tv",
        "--threshold_value",
        type=int,
        default=20,
        choices=range(0, 101),  # Allow values between 0 and 100
        metavar="[1-100]",
        help="HSV value threshold (in percent) used for the background detection.")
    parser.add_argument(
        '--denoise_median',
        action='store_true',
        default=False,
        help='Apply median denoising.')
    parser.add_argument(
        '--denoise_opening',
        action='store_true',
        default=False,
        help='Apply morphological opening on the (binary) background mask.')
    parser.add_argument(
        '--unsharp_mask',
        action='store_true',
        default=False,
        help='Apply unsharp masking with radius <-ur> and amount <-ua> to the V-channel of the output image in HSV representation.')
    parser.add_argument(
        "-ms",
        "--median_strength",
        type=int,
        default=3,
        help="Strength of median filtering")
    parser.add_argument(
        "-os",
        "--opening_strength",
        type=float,
        default=3,
        help="Strength of opening filtering / radius of the structuring element (disk)")
    parser.add_argument(
        "-ua",
        "--unsharp_amount",
        type=float,
        default=2,
        help="The amount used for unsharp masking.")
    parser.add_argument(
        "-ur",
        "--unsharp_radius",
        type=float,
        default=5,
        help="The radius used for unsharp masking.")
    parser.add_argument(
        '-k',
        '--keep_intermediate',
        action='store_true',
        default=False,
        help='Do not delete intermediate (single-page) PDFs afterwards.')
    parser.add_argument(
        '-v',
        '--verbose',
        action="count",
        default=0,
        help='Verbose output')
    parser.add_argument(
        '--version',
        action='version',
        version=VERSION,
        help='Show program version and exit')

    argcomplete.autocomplete(parser)
    return parser.parse_args()


def sort_filenames(filenames):
    """
    Sorts filenames in natural order.

    Args:
        filenames (list of str): The list of filenames.

    Returns:
        list of Path: The sorted list of filenames.
    """
    file_paths = [Path(file) for file in filenames]
    file_paths.sort(key=lambda path: natural_sort_key(path.name))
    return file_paths


def natural_sort_key(s):
    """
    Generate a key for natural sort.

    Args:
        s (str): The string to generate the key for.

    Returns:
        list: The key for natural sort.
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]


def sample_pixels(image, percentage):
    """
    Sample a given percentage of pixels in the image.

    Args:
        image (np.array): The image to sample pixels from.
        percentage (float): Percentage of pixels to sample

    Returns:
        np.array: The sampled pixels of shape
                    (N*M, 3) in case of color images
                    (N*M, 1) in case of grayscale and black-and-white images
    """
    if percentage == 100:
        if len(image.shape) == 2:
            return image.reshape((-1, 1))
        else:
            return image.reshape((-1, 3))

    else:
        total_pixels = image.shape[0] * image.shape[1]
        sample_size = int(total_pixels * (percentage / 100))
        rng = np.random.default_rng(0)
        indices = rng.choice(total_pixels, size=sample_size, replace=False)
        # black-and-white images
        if len(image.shape) == 2:
            return image.reshape((-1, 1))[indices]
        else:
            return image.reshape((-1, 3))[indices]


def perform_kmeans(pixels, n_clusters, args):
    """
    Performs k-means clustering on the image and return the cluster centers and the model.

    Args:
        image (np.array): The image to perform k-means clustering on.
                          Shape: (N*M, 3) or (N*M, 1)
        n_clusters (int): The number of clusters.
        args (argparse.Namespace): The command line arguments.

    Returns:
        tuple: The cluster centers and the fitted KMeans model.
    """
    assert pixels.shape[-1] in [1, 3]

    kmeans = KMeans(
        init = 'k-means++',
        n_init=1,
        n_clusters=n_clusters,
        copy_x = False,
        random_state=np.random.RandomState(0),
        verbose = True if args.verbose else False
        ).fit(pixels)

    return kmeans.cluster_centers_, kmeans


def pack_rgb(rgb):
    """
    Pack a 24-bit RGB triple into a single integer.

    Args:
        rgb (numpy.ndarray or tuple): The RGB values.

    Returns:
        int or numpy.ndarray: The packed RGB values.
    """

    if isinstance(rgb, np.ndarray):
        assert rgb.shape[-1] == 3  # RGB array must have 3 channels
        rgb = rgb.astype(np.uint32)
    else:
        assert len(rgb) == 3  # RGB tuple must have 3 channels
        rgb = np.array(rgb, dtype=np.uint32)

    packed = (rgb[:, 0] << 16 |
              rgb[:, 1] << 8 |
              rgb[:, 2])

    return packed


def unpack_rgb(packed):
    """
    Unpack a single integer or array of integers into one or more 24-bit RGB values.

    Args:
        packed (int or numpy.ndarray): The packed RGB values.

    Returns:
        numpy.ndarray: The unpacked RGB values.
    """

    return np.column_stack(((packed >> 16) & 0xff,
                            (packed >> 8) & 0xff,
                            packed & 0xff)).astype('uint8')


def rgb_to_sv(rgb):
    """
    Convert an RGB image or array of RGB colors to saturation and value, returning each one as a separate
    32-bit floating point array or value.

    Args:
        rgb (numpy.ndarray): The input RGB values.

    Returns:
        tuple: A tuple containing the saturation and value arrays or values.
    """

    if not isinstance(rgb, np.ndarray):
        rgb = np.array(rgb)

    rgb = rgb.reshape((-1, 3))

    rgb_min = np.nanmin(rgb, axis=1)
    rgb_max = np.nanmax(rgb, axis=1)

    # Avoid division by zero
    saturation = np.divide(rgb_max - rgb_min, rgb_max,
                           where=(rgb_max != 0),
                           out=np.zeros_like(rgb_max, dtype=float))

    value = rgb_max / 255.0

    return saturation, value


def create_palette(image_s, args, idx=None, use_global_palette=False):
    """
    Create a color palette from an image or a list of images.

    Args:
        image_s (np.array / list of filenames): Image file or list of image filenames if use_global_palette == True.
        args (argparse.Namespace): The command line arguments.
        use_global_palette (bool): Sample the color palette from a list of files instead of a single file

    Returns:
        tuple: The color palette and the fitted KMeans model.
    """

    if use_global_palette and isinstance(image_s, list):
        logging.info('Determining global color palette ...')
        sampled_pixels = [sample_pixels(io.imread(img), args.percentage) for img in image_s]

        # In case of mixed images types (color, grayscale, boolean / black-and-white),
        contains_bool  = any(arr.dtype == bool                           for arr in sampled_pixels)
        contains_gray  = any(arr.shape[-1] == 1 and arr.dtype == 'uint8' for arr in sampled_pixels)
        contains_color = any(arr.shape[-1] == 3                          for arr in sampled_pixels)

        if contains_color and (contains_gray or contains_bool):
            sampled_pixels = [np.array(Image.fromarray(arr).convert('RGB')) for arr in sampled_pixels
                                if arr.dtype == bool or (arr.shape[-1] == 1 and arr.dtype == 'uint8')]
            logging.warning('You have mixed color images with grayscale and / or black-and-white images while instructing to create a global palette.'
                +'This results in larger file sizes because all images are converted to color. Make sure this is what you want.')

        # Only color images
        elif contains_color and not (contains_gray or contains_bool):
            pass

        elif contains_gray and contains_bool:
            sampled_pixels = [np.array(Image.fromarray(arr).convert('L')) for arr in sampled_pixels
                                if arr.dtype == bool ]
            logging.warning('You have mixed grayscale images with black-and-white images while instructing to create a global palette.'
                +'This results in larger file sizes because all images are converted to grayscale. Make sure this is what you want.')

        # Only grayscale images
        elif contains_gray and not contains_bool:
            pass

        # Only black-and-white / boolean images
        else:
            pass

        sampled_pixels = np.vstack(sampled_pixels)

    elif not use_global_palette and isinstance(image_s, np.ndarray):
        logging.info('Page {}: Determining color palette ...'.format(idx))
        sampled_pixels = sample_pixels(image_s, args.percentage)

    else:
        raise RuntimeError('Programming error. Map map map map.')

    background_color = get_background_color(sampled_pixels)
    logging.info('Page {}:'.format(idx) if idx is not None else '' + 'Found background color {}'.format(background_color))

    if not background_color.dtype == bool:
        foreground_mask = get_foreground_mask(
            sampled_pixels,
            background_color=background_color,
            threshold_saturation=args.threshold_saturation,
            threshold_value=args.threshold_value)

        foreground_pixels = sampled_pixels[foreground_mask]

        logging.info('Page {}:'.format(idx) if idx is not None else '' + 'Clustering colors ...')
        kmeans_colors, kmeans_model = perform_kmeans(
            foreground_pixels, args.n_colors - 1, args)

        if args.white_background:
            color_palette = np.vstack(
                [[255, 255, 255] if len(sampled_pixels[0]) == 3 else [255], kmeans_colors]).round(0).astype('uint8')
        else:
            color_palette = np.vstack(
                [background_color, kmeans_colors]).round(0).astype('uint8')

        logging.info('Found {}colors: {}'.format('global ' if use_global_palette else '', color_palette.tolist()))

        return color_palette, kmeans_model

    else:
        return np.vstack([True, False]), None


def get_foreground_mask(
        image,
        background_color,
        threshold_saturation,
        threshold_value):
    """
    Get a binary mask of the foreground.

    Args:
        image (np.array): The image to get the foreground mask for.
                          Shape: (N*M, 3) or (N*M, 1)
        background_color (np.array): The background color.
        threshold_saturation (foat): The HSV Saturation threshold used for foreground-background discrimination.
        threshold_value (foat): The HSV value threshold used for foreground-background discrimination.

    Returns:
        np.array: The binary mask of the foreground.
    """
    assert image.shape[-1] in [1, 3]

    # color images
    if image.shape[-1] == 3:
        saturation_bg, value_bg = rgb_to_sv(background_color)
        saturation_image, value_image = rgb_to_sv(image)

        saturation_diff = np.abs(saturation_bg - saturation_image)
        value_diff = np.abs(value_bg - value_image)

        return ((value_diff >= (threshold_value / 100)) |
                (saturation_diff >= (threshold_saturation / 100)))

    # Grayscale images
    elif image.shape[-1] == 1 and image.dtype == 'uint8':
        value_diff = np.abs(background_color / 255. - image / 255.)
        return np.squeeze( value_diff >= (threshold_value / 100) )

    # black-and-white images
    else:
        raise RuntimeError('This should not happen.')

def quantize_colors(image, color_depth=6):
    """
    Reduce the number of colors in an image by reducing the number of bits per channel.

    Args:
        image (numpy.ndarray): The input image.
        color_depth (int): The number of bits per channel.

    Returns:
        numpy.ndarray: The quantized image.
    """

    assert image.dtype == np.uint8
    assert color_depth <= 8
    shift = 8 - color_depth

    # Truncate last shift bits and add half of the clipped bin
    return (image // 2**(shift)) * 2**(shift) + 2**(shift - 1)


def get_background_color(pixels, bits_per_channel=6):
    """
    Estimate the background color from an image or array of RGB colors by finding the most frequent color in the image.

    Args:
        pixels (numpy.ndarray): The RGB input pixels.
        bits_per_channel (int): The number of bits per channel.

    Returns:
        numpy.ndarray: An RGB tuple representing the background color.
                       [R, G, B] in case of color
                       [int] in case of grayscale
                       [True] in case of black-and-white (boolean)
    """

    # image was rolled out to shape (N*M, 3) (color) or (N*M, 1) (grayscale)
    assert pixels.shape[-1] in [1, 3]

    if pixels.shape[-1] == 3:
        quantized = quantize_colors(pixels, bits_per_channel).astype(np.uint32)
        packed = pack_rgb(quantized)

        unique, counts = np.unique(packed, return_counts=True)

        packed_background_color = unique[counts.argmax()]

        return unpack_rgb(packed_background_color)

    else:
        # Grayscale images
        if pixels.dtype == 'uint8':
            quantized = quantize_colors(pixels, bits_per_channel).astype(np.uint32)
            unique, counts = np.unique(quantized, return_counts=True)
            return np.array( [ unique[counts.argmax()] ], dtype='uint8' )

        elif pixels.dtype == bool:
            # background color is white in black-and-white images
            return np.array([True])


def apply_color_palette(image, color_palette, kmeans_model, args, idx):
    """
    Apply the color palette to the image using KMeans.predict.

    Args:
        image (np.array): The image of shape (N, M), (N, M, 1) or (N, M, 3) to apply the color palette to.
        color_palette (np.array): The color palette.
        kmeans_model (KMeans): The fitted KMeans model.
        args (argparse.Namespace): The command line arguments.

    Returns:
        np.array: The image with the color palette applied.
    """
    shape = image.shape
    assert len(shape) in [2, 3]

    if len(shape) == 3:
        image = image.reshape((-1, 3))
    else:
        image = image.reshape((-1, 1))

    if image.dtype == bool:
        foreground_mask = ~image.copy()

    else:
        foreground_mask = get_foreground_mask(
            image,
            background_color=color_palette[0],
            threshold_saturation=args.threshold_saturation,
            threshold_value=args.threshold_value)

    # morphological opening of the binary foreground mask to remove e.g. dust speckles
    if args.denoise_opening:
        logging.info('Page {}: Applying opening ...'.format(idx))
        # disk(<1) results in id-operation or zero-matrix and is hence useless
        kernel = disk(
            args.opening_strength) if args.opening_strength >= 1 else square(2)
        foreground_mask = binary_opening(
            foreground_mask.reshape(shape[:-1]), kernel).flatten()

    if image.dtype != bool:
        labels = np.zeros(image.shape[0], dtype='uint8')

        logging.info('Page {}: Applying color palette ...'.format(idx))
        # If the image is a solid color, the foreground has shape (0, 3)
        if image[foreground_mask].shape[0] != 0:
            labels[foreground_mask] = kmeans_model.predict(image[foreground_mask]) + 1

        if args.saturate:
            logging.info('Page {}: Maximizing saturation ...'.format(idx))
            color_palette = color_palette.astype(float)
            pmin = color_palette.min()
            pmax = color_palette.max()
            color_palette = 255 * (color_palette - pmin) / (pmax - pmin)
            color_palette = color_palette.round(0).astype('uint8')

        image = color_palette[labels]

    # set background to the background color
    image[~foreground_mask] = color_palette[0]

    if args.unsharp_mask:
        logging.info('Page {}: Applying unsharp mask filtering ...'.format(idx))

        if len(shape) == 3:
            image = np.array( Image.fromarray(image.reshape(shape), mode='RGB').convert('HSV') )

            # unsharp_mask returns in range [0, 1] and preserve_range seems broken
            tmp_max = np.max(image[:,:,2])
            image[:,:,2] = (unsharp_mask(image[:,:,2],
                                        radius=args.unsharp_radius,
                                        amount=args.unsharp_amount) * tmp_max).round(0).astype('uint8')
            image = np.array( Image.fromarray(image, mode='HSV').convert('RGB') ).reshape((-1,3))

        elif len(shape) == 2 and image.dtype != bool:
            tmp_max = np.max(image)
            image = (unsharp_mask(image.reshape(shape),
                                        radius=args.unsharp_radius,
                                        amount=args.unsharp_amount) * tmp_max).round(0).astype('uint8').reshape((-1,1))

        elif len(shape) == 2 and image.dtype == bool:
            logging.warning('Unsharp masking has no effect on binary / black-and-white images. Hence I skip this step ...')


    if args.denoise_median:
        logging.info('Page {}: Applying median filtering ...'.format(idx))
        # Median filtering is per color channel. In RGB space this would lead
        # to color deviations.
        if len(shape) == 3:
            image = Image.fromarray(image.reshape(shape), mode='RGB').convert('HSV')
            image = median_filter(
                image,
                size=(args.median_strength, args.median_strength, 1))
            image = Image.fromarray(image, mode='HSV').convert('RGB')

        else:
            image = median_filter(
                image.reshape(shape),
                size=(args.median_strength, args.median_strength))

        return np.array(image)

    else:
        return image.reshape(shape)


def generate_random_string(length=8):
    """
    Generate a random string of a given length consisting of ASCII letters and digits.

    Args:
        length (int): The length of the random string to be generated. Default is 8.

    Returns:
        str: A random string of the specified length.
    """
    letters = string.ascii_lowercase + string.digits
    return ''.join(random.choice(letters) for i in range(length))


def rename_with_random_string(filename):
    """
    Append a random string to the filename before the file extension.

    Args:
        filename (str): The original filename to be modified.

    Returns:
        Path: A new Path object with the modified filename.
    """
    path = Path(filename)
    random_str = generate_random_string()
    new_filename = f"{path.stem}-{random_str}{path.suffix}"
    return path.with_name(new_filename)


def handle_file_conflict(filename):
    """
    Handle file conflicts by renaming with a random string.

    In case of conflict, this function repeatedly generates a new filename by appending a random string
    until a filename is found that does not conflict with an existing file.

    Args:
        filename (str / pathlib.Path): The original filename that caused a conflict.

    Returns:
        filename (pathlib.Path): A new filename that does not conflict with existing files.
    """

    if filename.exists():
        while True:
            filename_new = rename_with_random_string(filename)
            if not filename_new.exists():
                logging.info(f"Renaming the file to: {filename_new}")
                return filename_new  # Return the non-conflicting filename
            else:
                logging.info("Random name still results in a conflict. Generating a new name.")
    else:
        return filename


def check_file_and_prompt(filename):
    """
    Check if a file exists and prompt the user to overwrite, exit, or rename the file.

    If the file exists, the user is prompted with options to overwrite (y), not overwrite (n),
    or rename (R) the file by appending a random string before the file extension.
    The default action for any other input is to rename the file.

    Args:
        filename (str / pathlib.Path): The filename to check and potentially modify.

    Returns:
        filename (str): The filename to use for the output file.
    """
    file_path = Path(filename)

    if file_path.exists():

        while True:
            choice = input('File "{}" exists. Overwrite? (y: yes; n: no and quit; r: rename) [ynR]: '.format(file_path)).strip().lower()

            if choice == 'y':
                print(f"Overwriting the file: {filename}")
                break

            elif choice == 'n':
                print("Not overwriting the file. Exiting program.")
                exit()

            elif choice == '' or choice == 'r':
                filename = handle_file_conflict(filename)
                break

    return filename


def save_as_pdf(image, filename, args, idx):
    """
    Saves a single <image> as a PDF <filename>.

    Args:
        image (np.array): The image to save
        filename (pathlib.Path): The output file
        args (argparse.Namespace): The command line arguments

    Returns:
        None
    """

    logging.info('Page {}: Saving page as {} ...'.format(idx, filename))
    pdf = Image.fromarray(image)
    pdf.save(filename, 'PDF',
             dpi=(args.dpi, args.dpi),
             quality=args.quality,
             optimize=True)


def merge_pdfs(filename_paths, args):
    """
    Merges multiple PDF files <filename_paths> into a single PDF file <args.output> using external ghostscript.

    Args:
        filename_paths (list of str / pathlib.Path): The list of the pdf file paths to merge.
        args (argparse.Namespace): The command line arguments

    Returns:
        None
    """
    filename = args.output

    if not args.overwrite:
        filename = check_file_and_prompt(filename)

    logging.info('Merging single pages to {} ...'.format(filename))
    try:
        command = ['gs',
                   '-dNOPAUSE',
                   '-sDEVICE=pdfwrite',
                   '-sOUTPUTFILE={}'.format(filename),
                   '-dBATCH'] + [str(f) for f in filename_paths]

        subprocess.run(command, check=True, capture_output=True)

    except subprocess.CalledProcessError as e:
        logging.critical(f"Error: {e}")
        logging.critical("Error: {}".format( e.stderr.decode('utf-8') if e.stderr is not None else None) )
        logging.critical("Stdout was: {}".format( e.stdout.decode('utf-8') if e.stdout is not None else None) )

    logging.info('Output written to {}'.format(args.output))


def process_image(file, output_filename, idx, args, global_palette=None):
    """
    Process a single image and save it as a PDF.

    This function performs the following steps:
    1. Reads the image from the specified file.
    2. Creates a color palette for the image or uses a global palette if provided.
    3. Applies the color palette to the image, potentially with saturation maximization and denoising.
    4. Saves the processed image as a PDF file with the given output filename.

    Args:
        file (pathlib.Path): The path to the input image file.
        output_filename (pathlib.Path): The path to the output PDF file.
        idx (int): The index of the image in the list of files being processed. Just used for the -v flag.
        args (argparse.Namespace): The command line arguments.
        global_palette (tuple, optional): A tuple containing the color palette and the fitted KMeans model,
                                          if using a global palette. Defaults to None.

    Returns:
        None
    """
    image = io.imread(file)

    processed_images = []

    logging.info('Processing image {}'.format(idx))

    if args.local_palette:
        color_palette, kmeans_model = create_palette(image, args, idx + 1)
    else:
        color_palette, kmeans_model = global_palette

    image = apply_color_palette(image, color_palette, kmeans_model, args, idx + 1)

    save_as_pdf(image, output_filename, args, idx + 1)


def check_file_existence(files):
    """
    Checks if a file / a list of files exist and raises an error if any are missing.

    Args:
        files (str / pathlib.Path or list of str / pathlib.Path): Files to be tested for existence.

    Raises:
        FileNotFoundError: If at least one file does not exist.
    """
    # Ensure <files> is a list.
    # In case <files> is neither str, nor Path, nor list, the error is raised by pathlib.Path below
    if not isinstance(files, list):
        files = [files]

    for file in files:
        path = Path(file)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file}")


def check_command_exists(command_name: str):
    """
    Checks if a given command exists in the system's PATH.

    Args:
        command_name (str): The name of the command to check.

    Returns:
        bool: True if the command is found in the PATH, False otherwise.
    """
    if not (shutil.which(command_name) is not None):
        raise RuntimeError('"{}" not found in PATH! Is it installed?'.format(command_name))


def main():
    """
    The main function of the program.
    """
    args = parse_args()

    if args.verbose == 1:
        logging.basicConfig(encoding='utf-8', format="%(levelname)s:%(message)s", level=logging.INFO)
    if args.verbose == 2:
        logging.basicConfig(encoding='utf-8', format="%(lineno)d-%(levelname)s:%(message)s", level=logging.DEBUG)

    check_command_exists('gs')

    file_paths = sort_filenames(args.files)
    check_file_existence(file_paths)

    # Create a temporary folder at the output file location for storing intermediate PDFs.
    # This way the intermediate files are automatically deleted upon program exit.
    # Each image is converted to a single-page PDF before concatenation
    # afterwards, which reduces the memory footprint.
    with tempfile.TemporaryDirectory(dir=os.getcwd(), prefix='tmp_pdfs-', delete=(not args.keep_intermediate)) as temp_dir:

        intermediate_pdf_paths = []

        if not args.local_palette:
            color_palette, kmeans_model = create_palette(file_paths, args, use_global_palette=True)

        with ThreadPoolExecutor(max_workers=args.jobs) as executor:

            threads = []
            for idx, file in enumerate(file_paths):

                output_filename = args.output.parent / temp_dir / Path(file.name).with_suffix('.pdf')

                # E.g. the same input file multiple times
                if output_filename in intermediate_pdf_paths or output_filename.exists():
                    output_filename = rename_with_random_string(output_filename)

                threads.append(executor.submit(process_image, file=file, output_filename=output_filename, idx=idx, args=args,
                                 global_palette=(color_palette, kmeans_model) if not args.local_palette else None))

                intermediate_pdf_paths.append(output_filename)

            executor.shutdown(wait=True)
            try:
                for f in threads:
                    f.result()  # This will raise a ValueError in case the thread crashed
            except ValueError as e:
                logging.critical(f"Caught an error: {e}")
                sys.exit(1)


        if args.keep_intermediate:
            logging.info('Skipping the deletion of intermediate PDFs (folder {}).'.format(temp_dir))

        merge_pdfs(intermediate_pdf_paths, args)


if __name__ == "__main__":
    main()
