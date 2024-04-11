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
python noteshrunk.py [-h] [-o OUTPUT] [-w] [-g] [-s] [-n N_COLORS] [-d DPI]
                    [-p PERCENTAGE] [-k] [-ts THRESHOLD_SATURATION]
                    [-tv THRESHOLD_VALUE] [--denoise_median] [--denoise_closing]
                    [--denoise_opening] [-ms MEDIAN_STRENGTH]
                    [-os OPENING_STRENGTH] [-cs CLOSING_STRENGTH] [-v] [-y]
                    files [files ...]
```

### Arguments

*   `files`: The input image files (supports various formats like PNG, JPG, etc.).
*   `-o, --output`: Path to the output PDF file (default: `output.pdf`).
*   `-w, --white_background`: Use a white background instead of the dominant color.
*   `-g, --global_palette`: Use the same color palette for all images.
*   `-s, --saturate`: Maximize saturation in the output image.
*   `-n, --n_colors`: Number of colors in the palette (default: 8).
*   `-d, --dpi`: DPI value for the input and output images (default: 300).
*   `-p, --percentage`: Percentage of pixels to sample from each image for palette creation (default: 10).
*   `-k, --keep_intermediate`: Keep the intermediate single-page PDFs.
*   `-ts, --threshold_saturation`: HSV saturation threshold for background detection (default: 15).
*   `-tv, --threshold_value`: HSV value threshold for background detection (default: 25).
*   `--denoise_median`: Apply median filtering for denoising.
*   `--denoise_closing`: Apply morphological closing for denoising.
*   `--denoise_opening`: Apply morphological opening for denoising.
*   `-ms, --median_strength`: Strength of median filtering (default: 3).
*   `-os, --opening_strength`: Strength of opening filtering (default: 3).
*   `-cs, --closing_strength`: Strength of closing filtering (default: 3).
*   `-v, --verbose`: Enable verbose output.
*   `-y, --overwrite`: Overwrite existing files without prompting.

## Examples

1.  Compress a single image with default settings:

    ```bash
    noteshrunk input.png
    ```

2.  Compress multiple images with a white background and 16 colors:

    ```bash
    noteshrunk -w -n 16 image1.jpg image2.png
    ```

3.  Compress images using a global palette and keep intermediate files:
    ```bash
    noteshrunk -g -k *.jpg
    ```

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests on the GitHub repository.

## Acknowledgements

This project utilizes open-source software from the Python community.
Special thanks to the developers and maintainers of the required libraries as well as [mzucker's](https://github.com/mzucker/noteshrink) initial program.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
