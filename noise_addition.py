import os
import numpy as np
import cv2
from PIL import Image

# ------------------------------
# Noise Functions (With Optimized Parameters)
# ------------------------------

def add_gaussian_noise(image_array, mean=0, sigma=15):
    """Adds Gaussian noise with optimized sigma."""
    gaussian_noise = np.random.normal(mean, sigma, image_array.shape)
    noisy_image = image_array + gaussian_noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def add_salt_and_pepper_noise(image_array, salt_prob=0.02, pepper_prob=0.02):
    """Adds Salt-and-Pepper noise with adjusted probability."""
    noisy_image = np.copy(image_array)
    rand_matrix = np.random.rand(*image_array.shape[:2])  # Shape: (H, W)

    for c in range(image_array.shape[2]):  # Iterate over RGB channels
        noisy_image[rand_matrix < salt_prob, c] = 255  # White (Salt)
        noisy_image[rand_matrix > 1 - pepper_prob, c] = 0  # Black (Pepper)

    return noisy_image

def add_speckle_noise(image_array, intensity=0.1):
    """Adds Speckle noise with reduced intensity."""
    noise = np.random.randn(*image_array.shape) * intensity
    noisy_image = image_array + image_array * noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def add_poisson_noise(image_array):
    """Adds Poisson noise while ensuring proper image intensity scaling."""
    image_array = image_array.astype(np.float32)
    noisy_image = np.random.poisson(image_array * 255) / 255.0
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def add_uniform_noise(image_array, low=-20, high=20):
    """Adds Uniform noise with lower range for realistic degradation."""
    uniform_noise = np.random.uniform(low, high, image_array.shape)
    noisy_image = image_array + uniform_noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def add_jpeg_compression_noise(image_array, quality=60):
    """Applies JPEG compression noise with optimized quality."""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded_image = cv2.imencode('.jpg', image_array, encode_param)  # Compress
    decoded_image = cv2.imdecode(encoded_image, cv2.IMREAD_UNCHANGED)  # Decompress
    return decoded_image

# ------------------------------
# Processing Function
# ------------------------------

def process_image(input_path, output_folder, noise_functions, jpeg_quality=60):
    """Applies different noise types, then JPEG compression, and saves as PNG."""
    try:
        with Image.open(input_path) as img:
            img = img.convert("RGB")  # Ensure image is in RGB format
            image_array = np.array(img)
    except Exception as e:
        print(f"⚠ Error opening {input_path}: {e}")
        return

    base_name, _ = os.path.splitext(os.path.basename(input_path))

    for noise_name, noise_func in noise_functions.items():
        try:
            # Apply individual noise
            noisy_array = noise_func(image_array)

            # Apply JPEG compression noise after the individual noise
            noisy_array = add_jpeg_compression_noise(noisy_array, quality=jpeg_quality)

            # Save in a dedicated folder
            noise_folder = os.path.join(output_folder, noise_name)
            os.makedirs(noise_folder, exist_ok=True)
            output_path = os.path.join(noise_folder, f"{base_name}.png")

            noisy_img = Image.fromarray(noisy_array)
            noisy_img.save(output_path)
            print(f"✔ Saved: {output_path}")

        except Exception as e:
            print(f"⚠ Error processing {base_name} with {noise_name} noise: {e}")

# ------------------------------
# Batch Processing for a Folder
# ------------------------------

def process_folder(input_folder, output_base_folder, noise_functions, jpeg_quality=60):
    """Processes all images in a folder, applying each noise type and JPEG compression."""
    os.makedirs(output_base_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            process_image(os.path.join(input_folder, filename), output_base_folder, noise_functions, jpeg_quality)

    print("✅ All images processed and saved!")

# ------------------------------
# Main Execution
# ------------------------------

if __name__ == "__main__":
    INPUT_FOLDER = "./image_dataset/256x256"
    OUTPUT_BASE_FOLDER = "./testing/noisy_images/256x256"

    noise_functions = {
        "Gaussian": add_gaussian_noise,
        "SaltPepper": add_salt_and_pepper_noise,
        "Speckle": add_speckle_noise,
        "Poisson": add_poisson_noise,
        "Uniform": add_uniform_noise
    }

    process_folder(INPUT_FOLDER, OUTPUT_BASE_FOLDER, noise_functions, jpeg_quality=60)
