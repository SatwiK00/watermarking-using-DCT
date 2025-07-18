import os
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from skimage.restoration import denoise_tv_chambolle
from numpy.lib.stride_tricks import as_strided

def get_optimal_block_size(image_shape):
    """Determines the optimal block size based on image resolution."""
    height, width = image_shape[:2]  # Modified for color images
    min_dim = min(height, width)

    if min_dim >= 2048:
        return 128
    elif min_dim >= 1024:
        return 64
    elif min_dim >= 512:
        return 32
    else:
        return 16  # Minimum block size for small images

def pad_image_to_block_size(image, block_size):
    """Pads image so that its dimensions are multiples of block size."""
    h, w = image.shape
    pad_h = (block_size - (h % block_size)) % block_size
    pad_w = (block_size - (w % block_size)) % block_size

    return np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')

def extract_watermark(image, block_size, watermark_pos=(4, 3)):
    """Extracts watermark from all RGB channels and aggregates them."""
    watermarks = []
    
    # Process each channel separately
    for channel in cv2.split(image):  # Split into B, G, R channels
        channel_padded = pad_image_to_block_size(channel, block_size)
        h, w = channel_padded.shape

        # Reshape into blocks and compute DCT
        shape = (h // block_size, w // block_size, block_size, block_size)
        strides = (
            block_size * channel_padded.strides[0],
            block_size * channel_padded.strides[1],
            channel_padded.strides[0],
            channel_padded.strides[1],
        )
        blocks = as_strided(channel_padded, shape=shape, strides=strides)
        dct_blocks = np.array([[cv2.dct(np.float32(block)) for block in row] for row in blocks])

        # Extract coefficients from current channel
        wm_channel = dct_blocks[:, :, watermark_pos[0], watermark_pos[1]]
        watermarks.append(wm_channel)

    # Average across channels to improve robustness
    aggregated_watermark = np.mean(watermarks, axis=0)
    
    # Normalize and return
    return np.uint8(cv2.normalize(aggregated_watermark, None, 0, 255, cv2.NORM_MINMAX))

def apply_denoising_filter(watermark, noise_type):
    """
    Applies noise-specific denoising filters.
    """
    noise_type = noise_type.lower()

    if "gaussian" in noise_type:
        return cv2.GaussianBlur(watermark, (5, 5), 0)
    elif "salt" in noise_type or "pepper" in noise_type:
        return cv2.medianBlur(watermark, 3)
    elif "speckle" in noise_type:
        return cv2.bilateralFilter(watermark, 9, 75, 75)
    elif "poisson" in noise_type or "uniform" in noise_type:
        return cv2.fastNlMeansDenoising(watermark, None, h=10, templateWindowSize=7, searchWindowSize=21)
    else:
        return watermark.copy()

def process_noisy_watermarks(noisy_images, output_folder, watermark_pos=(4,3)):
    """
    Processes noisy images in parallel to extract watermarks and save them.
    """
    def process_image(resolution, noise_type, file):
        noise_path = os.path.join(noisy_images, resolution, noise_type, file)
        if not os.path.isfile(noise_path):
            return
        
        # Read as COLOR image instead of grayscale
        noisy_img = cv2.imread(noise_path, cv2.IMREAD_COLOR)  # Changed to color
        if noisy_img is None:
            print(f"Skipping {file}: unable to load image.")
            return

        # Use shape[:2] to ignore color channels
        block_size = get_optimal_block_size(noisy_img.shape[:2])  # Modified here
        watermark = extract_watermark(noisy_img, block_size, watermark_pos)
        watermark_denoised = apply_denoising_filter(watermark, noise_type)

        out_path = os.path.join(output_folder, resolution, noise_type, file)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, watermark_denoised)
        print(f"✔ Saved: {out_path}")

    with ThreadPoolExecutor() as executor:
        for resolution in os.listdir(noisy_images):
            for noise_type in os.listdir(os.path.join(noisy_images, resolution)):
                for file in os.listdir(os.path.join(noisy_images, resolution, noise_type)):
                    executor.submit(process_image, resolution, noise_type, file)

if __name__ == "__main__":
    noisy_images = r"./noisy_images"
    output_folder = r"./noisy_watermarks"

    process_noisy_watermarks(noisy_images, output_folder)