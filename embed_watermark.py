import os
import cv2
import numpy as np
from scipy.fftpack import dct, idct

# ------------------------------
# Helper Functions
# ------------------------------

def apply_dct(block):
    """ Apply Discrete Cosine Transform (DCT) """
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def apply_idct(block):
    """ Apply Inverse Discrete Cosine Transform (IDCT) """
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

def get_block_size(image_size):
    """ Determine the optimal block size based on image size """
    if image_size <= 256:
        return 16
    elif image_size <= 512:
        return 32
    elif image_size <= 1024:
        return 64
    else:
        return 128  # For 2048x2048 images

def get_optimal_alpha(image_size):    #(balance between quality and compression efficiency)
    """ Dynamically adjust alpha for different resolutions """
    if image_size <= 256:
        return 5  # Lower alpha for small images(fine details matter)
    elif image_size <= 512:
        return 10
    elif image_size <= 1024:
        return 15
    else:
        return 25  # Higher alpha for large images(processing speed matters)

# ------------------------------
# Watermark Embedding Function
# ------------------------------

def embed_watermark_color(image, watermark):
    """
    Embed a watermark in all three color channels (R, G, B) separately.

    Parameters:
        image (numpy.ndarray): Original color image (BGR format).
        watermark (numpy.ndarray): Watermark image.

    Returns:
        watermarked_image (numpy.ndarray): Image with embedded watermark.
    """
    h, w, _ = image.shape
    block_size = get_block_size(h)  # Auto-select block size
    alpha = get_optimal_alpha(h)  # Dynamically tuned alpha

    # Ensure image dimensions are multiples of block size
    h_new = (h // block_size) * block_size
    w_new = (w // block_size) * block_size
    image = cv2.resize(image, (w_new, h_new))

    # Resize watermark to match the new block grid
    wm_resized = cv2.resize(watermark, (w_new // block_size, h_new // block_size))
    _, wm_binary = cv2.threshold(wm_resized, 127, 1, cv2.THRESH_BINARY)

    # Split channels and process separately
    b, g, r = cv2.split(image)
    b_wm = embed_watermark(b, wm_binary, alpha, block_size)
    g_wm = embed_watermark(g, wm_binary, alpha, block_size)
    r_wm = embed_watermark(r, wm_binary, alpha, block_size)

    return cv2.merge([b_wm, g_wm, r_wm])

def embed_watermark(channel, watermark, alpha, block_size):
    """
    Embed the watermark into a single color channel using adaptive DCT block size.

    Parameters:
        channel (numpy.ndarray): Single-channel grayscale image.
        watermark (numpy.ndarray): Binary watermark.
        alpha (float): Embedding strength.
        block_size (int): DCT block size.

    Returns:
        watermarked_channel (numpy.ndarray): Watermarked grayscale channel.
    """
    h, w = channel.shape
    watermarked_channel = np.zeros_like(channel, dtype=np.float32)

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = channel[i:i+block_size, j:j+block_size].astype(np.float32)
            dct_block = apply_dct(block)

            # Get corresponding watermark bit
            wm_x = min(i // block_size, watermark.shape[0] - 1)
            wm_y = min(j // block_size, watermark.shape[1] - 1)
            wm_bit = watermark[wm_x, wm_y]

            # Embed watermark at mid-frequency location (adaptive block size)   
            #best balance between visibility,robustness and secuirty to attacks
            dct_block[block_size//2, block_size//4] += alpha * wm_bit  

            watermarked_channel[i:i+block_size, j:j+block_size] = apply_idct(dct_block)

    return np.clip(watermarked_channel, 0, 255).astype(np.uint8)

# ------------------------------
# Process Images in Folder
# ------------------------------

def process_images(input_folder, output_folder, watermark_path):
    """
    Process all images in a folder and embed the watermark dynamically.

    Parameters:
        input_folder (str): Path to the folder containing images.
        output_folder (str): Path to save watermarked images.
        watermark_path (str): Path to the watermark image.
    """
    watermark = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
    if watermark is None:
        print("Error: Could not load watermark image!")
        return

    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            image = cv2.imread(input_path)
            if image is None:
                print(f"Skipping {filename}: Unable to read.")
                continue

            print(f"Processing {filename}...")

            # Embed watermark
            watermarked_image = embed_watermark_color(image, watermark)

            # Save the output image
            cv2.imwrite(output_path, watermarked_image, [cv2.IMWRITE_JPEG_QUALITY, 95])

    print("Watermarking process completed!")

# ------------------------------
# Main Execution
# ------------------------------

if __name__ == "__main__":
    input_folder = "./testing/test_directory/512x256"         # Folder with images
    output_folder = "./testing/watermarked_images/256x256"          # Output folder
    watermark_path = "./watermarks/watermark2048.png"  # Watermark path

    process_images(input_folder, output_folder, watermark_path)
