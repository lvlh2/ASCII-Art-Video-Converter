#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File: ascii_art.py
@Time: 2024/10/07 19:17:55
@Author: lvlh2
"""


import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# * The rate of downsampling.
SAMPLE_RATE = 0.1
# * Compresses the image as the spaces between symbols are too large.
COMPRESS_RATIO = 0.6

RAW_VIDEO = 'video.mp4'
OUTPUT_VIDEO = 'ascii_video.mp4'

# * The folder for the raw extract frames.
RAW_IMAGE_FOLDER = 'raw images'
# * The folder for the converted ascii images.
ASCII_IMAGE_FOLDER = 'ascii images'

MAX_WORKERS = 20


class AsciiArtist:
    """A tool to convert a video to an ascii video."""

    def extract_frames(self, video: str, output_folder: str) -> None:
        """Extracts all the frames from the video.

        Args:
            video (str): The path of the video.
            output_folder (str): The path of the output folder.
        """
        # Extracts frames from the video.
        cmd = [
            'ffmpeg',
            '-i',
            video,
            '-r',
            '25',
            '-qscale:v',
            '2',
            f'{output_folder}/%04d.jpg',
        ]
        subprocess.run(cmd, check=True)

    def convert_to_ascii(self, image: str) -> None:
        """Converts the image to an ascii image.

        Args:
            image (str): The file name of the image.
        """
        im = Image.open(f'{RAW_IMAGE_FOLDER}/{image}')

        # Downsamples the raw image.
        new_size = np.array(
            [im.width * SAMPLE_RATE, im.height * SAMPLE_RATE], dtype='int'
        )
        im_resized = im.resize(tuple(new_size))

        # Gets the colors for ascii symbols.
        color = np.array(im_resized)

        # Converts to grayscale
        im_gray = im_resized.convert('L')

        # Generates the indices for the symbols based on the grayscale.
        gray = np.array(im_gray)
        symbols = list('!v+#S%$&NM@')
        symbol_indices = ((gray / 255) * (len(symbols) - 1)).astype('int')

        # Gets the width and height of the default font.
        font = ImageFont.load_default(size=30)
        box = font.getbbox('x')
        font_width = box[2] - box[0]
        font_height = box[3] - box[1]
        font_size = np.array([font_width, font_height], dtype='int')

        # The size of the output image
        # which is the number of symbols (`new_size`) multiplied by the font size.
        out_size = new_size * font_size

        # Compresses the image as the spaces between symbols are too large.
        compress_ratio = 0.6
        out_size = (compress_ratio * out_size).astype('int')

        # Makes sure the height can be divided by 2,
        # which is demanded by FFmpeg when converting images to a video.
        if out_size[1] % 2 != 0:
            out_size[1] += 1

        im_out = Image.new(
            mode='RGB',
            size=tuple(out_size),
            color='black',
        )
        draw = ImageDraw.Draw(im_out)

        # Draws the symbols.
        for i in range(new_size[0]):
            for j in range(new_size[1]):
                draw.text(
                    (
                        # The spaces between symbols are compressed.
                        compress_ratio * font_size[0] * i,
                        compress_ratio * font_size[1] * j,
                    ),
                    # * The indices of `numpy.array` is reversed in terms of rows and columns,
                    # * thus `[j, i]`.
                    text=symbols[symbol_indices[j, i]],
                    fill=tuple(color[j, i]),
                )

        im_out.save(f'{ASCII_IMAGE_FOLDER}/{image}')

    def convert_images_to_video(self, image_folder: str) -> None:
        """Converts the sequence of ascii images to a video.

        Args:
            image_folder (str): The path of the folder holding the ascii images.
        """
        cmd = [
            'ffmpeg',
            '-i',
            f'{image_folder}/%04d.jpg',
            '-c:v',
            'libx264',
            '-vf',
            'fps=25',
            '-pix_fmt',
            'yuv420p',
            '-y',
            OUTPUT_VIDEO,
        ]
        subprocess.run(cmd, check=True)


def main():
    path = os.path.dirname(__file__)
    os.chdir(path)

    # Makes folders for raw images and ascii images.
    for folder in [RAW_IMAGE_FOLDER, ASCII_IMAGE_FOLDER]:
        if os.path.exists(folder):
            shutil.rmtree(folder)

        os.makedirs(folder)

    artist = AsciiArtist()

    # Extracts images from the raw video.
    artist.extract_frames(video=RAW_VIDEO, output_folder=RAW_IMAGE_FOLDER)

    images = os.listdir(RAW_IMAGE_FOLDER)
    # Converts raw images to ascii images with thread pooling.
    with ThreadPoolExecutor(MAX_WORKERS) as executor:
        futures = [
            executor.submit(artist.convert_to_ascii, image=image) for image in images
        ]
        for i, future in enumerate(as_completed(futures), start=1):
            try:
                future.result()
                print(f'Progress: {i} of {len(images)}.')
            except Exception as e:
                print(e)

    # Converts the ascii images to a video.
    artist.convert_images_to_video(image_folder=ASCII_IMAGE_FOLDER)


if __name__ == '__main__':
    t0 = time.time()
    main()
    t1 = time.time()
    print(f'Total time: {t1 - t0:.2f} s.')
