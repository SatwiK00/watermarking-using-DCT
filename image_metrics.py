import os
import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from concurrent.futures import ThreadPoolExecutor, as_completed

# Directories
ORIGINAL_DIR = "./image_dataset"
NOISY_DIR = "./noisy_images"
OUTPUT_CSV = "./image_metrics.csv"

def resize_image(image, target_shape):
    """Resizes the image to match the target shape."""
    return cv2.resize(image, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_CUBIC)

def convert_to_png(image):
    """Converts an image to PNG format in memory and decodes it back as grayscale."""
    is_success, buffer = cv2.imencode(".png", image)
    if is_success:
        return cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)
    else:
        print("[⚠] Warning: Conversion to PNG failed.")
        return image

def calculate_psnr(original, noisy):
    """Computes PSNR (Peak Signal-to-Noise Ratio) between original and noisy images."""
    mse = np.mean((original - noisy) ** 2)
    eps = 1e-10  # To prevent division by zero
    return 10 * np.log10((255 ** 2) / (mse + eps))

def calculate_correlation(original, noisy):
    """Computes the correlation coefficient between the original and noisy images."""
    orig_flat = original.flatten()
    noisy_flat = noisy.flatten()
    corr_matrix = np.corrcoef(orig_flat, noisy_flat)
    return corr_matrix[0, 1]

def calculate_ssim(original, noisy):
    """Computes SSIM (Structural Similarity Index Measure) between original and noisy images."""
    return ssim(original, noisy, data_range=255)

def process_single_image(resolution, noise_type, noisy_file, original_file_path, noisy_file_path):
    """Processes a single noisy image by comparing it with its matching original image."""
    # Load original image in color, then convert to grayscale
    original = cv2.imread(original_file_path, cv2.IMREAD_COLOR)
    if original is None:
        print(f"[⚠] Unable to load original image: {original_file_path}")
        return None

    original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    original = convert_to_png(original)

    # Load the noisy image in color, then convert to grayscale
    noisy = cv2.imread(noisy_file_path, cv2.IMREAD_COLOR)
    if noisy is None:
        print(f"[⚠] Unable to load noisy image: {noisy_file_path}")
        return None

    noisy = cv2.cvtColor(noisy, cv2.COLOR_BGR2GRAY)
    # Resize the noisy image to the original's dimensions
    noisy_resized = resize_image(noisy, original.shape)

    # Compute metrics
    psnr_value = calculate_psnr(original, noisy_resized)
    ssim_value = calculate_ssim(original, noisy_resized)
    correlation_value = calculate_correlation(original, noisy_resized)

    print(f"[✓] Processed {noisy_file} | PSNR: {psnr_value:.2f} dB | SSIM: {ssim_value:.4f} | Corr: {correlation_value:.4f}")
    return [resolution, noise_type, noisy_file, psnr_value, ssim_value, correlation_value]

def process_images():
    results = []

    # Build a list of tasks
    tasks = []
    with ThreadPoolExecutor() as executor:
        # Loop over each resolution folder (e.g., 256x256, 512x512, etc.)
        for resolution in os.listdir(ORIGINAL_DIR):
            orig_res_path = os.path.join(ORIGINAL_DIR, resolution)
            noisy_res_path = os.path.join(NOISY_DIR, resolution)
            
            if not os.path.isdir(orig_res_path) or not os.path.isdir(noisy_res_path):
                continue

            # Build a mapping for original images: base filename -> full path
            original_images = {}
            for file in os.listdir(orig_res_path):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    base = os.path.splitext(file)[0]
                    original_images[base] = os.path.join(orig_res_path, file)

            # For each noise type folder inside the resolution folder
            for noise_type in os.listdir(noisy_res_path):
                noise_type_path = os.path.join(noisy_res_path, noise_type)
                if not os.path.isdir(noise_type_path):
                    continue
                
                for file in os.listdir(noise_type_path):
                    if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                        continue

                    base = os.path.splitext(file)[0]
                    if base not in original_images:
                        print(f"[⚠] No matching original for {file} in {resolution}/{noise_type}")
                        continue

                    original_file_path = original_images[base]
                    noisy_file_path = os.path.join(noise_type_path, file)
                    # Submit task to the executor
                    tasks.append(executor.submit(
                        process_single_image,
                        resolution,
                        noise_type,
                        file,
                        original_file_path,
                        noisy_file_path
                    ))

        # Retrieve task results as they complete
        for future in as_completed(tasks):
            result = future.result()
            if result is not None:
                results.append(result)

    # Save results to CSV
    df = pd.DataFrame(results, columns=["Resolution", "Noise Type", "Image Name", "PSNR", "SSIM", "Correlation"])
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[✔] Metrics saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    process_images()
