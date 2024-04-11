# noteshrunk - Document Color Palette Compression

This Python script compresses images by reducing the number of colors and optimizing the image representation.
It leverages KMeans clustering for color quantization and offers various options to customize the compression process.
All supplied images are then saved as a multi-page PDF.

The idea of the program is to optimize scanned documents.
This is a complete and improved rewrite of [mzucker's](https://github.com/mzucker/noteshrink) noteshrink.

## Features

* **Color Quantization:** Reduces the number of colors in the document using KMeans clustering, leading to smaller file sizes.
* **Background Detection and Removal:** Identifies and removes the background color.
* **Customizable Palette:** Allows you to specify the number of colors in the output palette and choose between a global palette for all pages or individual palettes for each page.
* **Color Control:** Offers the option to maximize saturation in the output image as well as to remove the background (replace with white), enhancing visual clarity.
* **Denoising Options:** Provides median filtering and morphological operations to reduce noise and improve image quality.
* **Parallel processing:** Utilizes multiple CPU cores for faster processing of multiple images.

## Requirements

- argcomplete
- NumPy
- Pillow (PIL Fork)
- Python 3
- scikit-image
- scikit-learn
- SciPy

### Optional

- Ghostscript (for PDF merging; otherwise you need to use the `-k` flag)

## Installation

```bash
pipx install noteshrunk
```

## Usage

```
noteshrunk [-h] [-o OUTPUT] [-w] [-g] [-s] [-n N_COLORS] [-d DPI] [-q [1-100]]
           [-p PERCENTAGE] [-k] [-ts THRESHOLD_SATURATION] [-tv THRESHOLD_VALUE]
           [--denoise_median] [--denoise_closing] [--denoise_opening] [-ms MEDIAN_STRENGTH]
           [-os OPENING_STRENGTH] [-cs CLOSING_STRENGTH] [-j JOBS] [-v] [-y]
           files [files ...]
```

### Arguments

* `files`: A list of paths to the input image files.
* `-o`, `--output`: Path to the output PDF file (default: `output.pdf`).
* `-w`, `--white_background`: Use white background instead of dominant color.
* `-g`, `--global_palette`: Use the same color palette for all images by sampling a percentage of the pixels from every input image.
* `-s`, `--saturate`: Maximize saturation in the output image.
* `-n`, `--n_colors`: Number of colors in the palette (default: 8).
* `-d`, `--dpi`: DPI value of the input images (default: 300).
* `-q`, `--quality`: JPEG quality of the embedded images (1-100, default: 75).
* `-p`, `--percentage`: Percentage of pixels to sample from every image for global palette creation (default: 10).
* `-k`, `--keep_intermediate`: Keep the intermediate single-page PDFs.
* `-ts`, `--threshold_saturation`: HSV saturation threshold (in percent) used for background detection (default: 15).
* `-tv`, `--threshold_value`: HSV value threshold (in percent) used for background detection (default: 25).
* `--denoise_median`: Apply median denoising.
* `--denoise_closing`: Apply morphological closing on the background mask.
* `--denoise_opening`: Apply morphological opening on the background mask.
* `-ms`, `--median_strength`: Strength of median filtering (default: 3).
* `-os`, `--opening_strength`: Strength of opening filtering / radius of the structuring element (disk, default: 3).
* `-cs`, `--closing_strength`: Strength of closing filtering / radius of the structuring element (disk, default: 3).
* `-j`, `--jobs`: Number of processes to use (default: number of CPU cores).
* `-v`, `--verbose`: Verbose output.
* `-y`, `--overwrite`: Overwrite existing files without asking.

## Examples

1.  Compress a single image with default settings:

    ```bash
    noteshrunk input.png
    ```

2.  Compress multiple images with a white background and 16 colors:

    ```bash
    noteshrunk -w -n 16 image1.jpg image2.png
    ```

3.  Compress images using a global palette and keep intermediate files while disabling multi-processing:
    ```bash
    noteshrunk -g -j 1 -k *.jpg
    ```

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests on the GitHub repository.

## Acknowledgements

This project utilizes open-source software from the Python community.
Special thanks to the developers and maintainers of the required libraries as well as [mzucker's](https://github.com/mzucker/noteshrink) initial program.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
