import os
import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from concurrent.futures import ThreadPoolExecutor

# Mapping resolution to the corresponding original watermark file
WATERMARK_FILES = {
    "256x256": "watermark256.png",
    "512x512": "watermark512.png",
    "1024x1024": "watermark1024.png",
    "2048x2048": "watermark2048.png"
}

def resize_image(image, target_shape):
    """Resizes the image to match the target shape."""
    return cv2.resize(image, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_CUBIC)

def calculate_psnr(original, noisy):
    """Computes PSNR (Peak Signal-to-Noise Ratio) between original and noisy images."""
    mse = np.mean((original - noisy) ** 2)
    eps = 1e-10  # Small value to prevent divide-by-zero errors
    return 10 * np.log10((255 ** 2) / (mse + eps))

def calculate_correlation(original, noisy):
    """Computes the correlation coefficient between the original and noisy images."""
    original_flatten = original.flatten()
    noisy_flatten = noisy.flatten()
    correlation_matrix = np.corrcoef(original_flatten, noisy_flatten)
    return correlation_matrix[0, 1]  # Extract the correlation value

def calculate_ssim(original, noisy):
    """Computes SSIM (Structural Similarity Index Measure) between original and noisy images."""
    return ssim(original, noisy, data_range=255)

def process_single_image(resolution, noise_type, file, original, noise_path, results):
    """Processes a single noisy watermark image and computes metrics."""
    noisy_img_path = os.path.join(noise_path, file)
    noisy = cv2.imread(noisy_img_path, cv2.IMREAD_GRAYSCALE)
    
    if noisy is None:
        print(f"[⚠] Skipping {file}: Unable to load image.")
        return

    noisy_resized = resize_image(noisy, original.shape)

    # Compute metrics
    psnr_value = calculate_psnr(original, noisy_resized)
    ssim_value = calculate_ssim(original, noisy_resized)
    correlation_value = calculate_correlation(original, noisy_resized)

    # Append results
    results.append([resolution, noise_type, file, psnr_value, ssim_value, correlation_value])
    
    print(f"[✓] Processed: {file} | PSNR: {psnr_value:.2f} dB | SSIM: {ssim_value:.4f} | Correlation: {correlation_value:.4f}")

def process_images(original_watermark_folder, noisy_watermark_folder, output_csv):
    """
    Compares extracted noisy watermarks with their respective original watermarks, calculates metrics, and saves them to CSV.
    
    Args:
        original_watermark_folder: Folder containing original watermark images.
        noisy_watermark_folder: Folder containing noisy watermarks.
        output_csv: Output CSV file to store results.
    """
    results = []

    with ThreadPoolExecutor() as executor:
        for resolution in os.listdir(noisy_watermark_folder):
            res_path = os.path.join(noisy_watermark_folder, resolution)
            if not os.path.isdir(res_path) or resolution not in WATERMARK_FILES:
                continue  # Skip if not a valid resolution

            # Load the corresponding original watermark
            original_watermark_path = os.path.join(original_watermark_folder, WATERMARK_FILES[resolution])
            original = cv2.imread(original_watermark_path, cv2.IMREAD_GRAYSCALE)
            if original is None:
                print(f"[⚠] Error: Cannot load original watermark for resolution {resolution} from {original_watermark_path}")
                continue

            for noise_type in os.listdir(res_path):
                noise_path = os.path.join(res_path, noise_type)
                if not os.path.isdir(noise_path):
                    continue

                for file in os.listdir(noise_path):
                    if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                        continue  # Skip non-image files

                    executor.submit(process_single_image, resolution, noise_type, file, original, noise_path, results)

    # Save results to CSV
    df = pd.DataFrame(results, columns=["Resolution", "Noise Type", "Image Name", "PSNR", "SSIM", "Correlation"])
    df.to_csv(output_csv, mode='a', header=not os.path.exists(output_csv), index=False)
    print(f"\n[✔] Results saved to {output_csv}")

if __name__ == "__main__":
    original_watermark_folder = "./watermarks"  # Folder containing original watermarks
    noisy_watermark_folder = "./noisy_watermarks"  # Folder containing noisy extracted watermarks
    output_csv = "./watermark_metrics.csv"  # Output CSV file

    process_images(original_watermark_folder, noisy_watermark_folder, output_csv)
