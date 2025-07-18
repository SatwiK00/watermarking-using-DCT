import os
import requests
from PIL import Image
from io import BytesIO
from pexels_api import API

# ✅ Pexels API Key (Replace with your own key)
PEXELS_API_KEY = "mmA0BUNSrFyDjgX09Sew6KAyrgLup4yDUWt8m8gW2sVOCHcskk8Kyj0k"
api = API(PEXELS_API_KEY)

# ✅ Search parameters
SEARCH_QUERY = "nature"  # Change keyword
NUM_IMAGES = 904  # Total images to download
TARGET_RESOLUTION = (2048, 2048)  # Resize to (width, height)
RESULTS_PER_PAGE = 80  # Pexels API max per page

# ✅ Folder to save images
SAVE_FOLDER = "./watermark_dataset/2048x2048"  # Change folder
os.makedirs(SAVE_FOLDER, exist_ok=True)

def get_filename_from_url(url):
    """Extracts filename from image URL"""
    return url.split("/")[-1].split("?")[0]

def download_images():
    """Fetches multiple pages to download more images"""
    
    print(f"\n🔍 Searching for '{SEARCH_QUERY}' images...\n")

    downloaded_count = 0
    existing_files = set(os.listdir(SAVE_FOLDER))  # Check already downloaded images

    page = 1  # Start from page 1
    while downloaded_count < NUM_IMAGES:
        api.search(SEARCH_QUERY, page=page, results_per_page=RESULTS_PER_PAGE)
        photos = api.get_entries()
        
        if not photos:
            print("❌ No more images found. Stopping search.")
            break  # Stop if no more images available

        for photo in photos:
            if downloaded_count >= NUM_IMAGES:
                break  # Stop when enough images are downloaded

            img_url = photo.original
            filename = get_filename_from_url(img_url)

            if filename in existing_files:
                print(f"⏩ Skipping (Already Exists): {filename}")
                continue  # Skip duplicate image

            save_path = os.path.join(SAVE_FOLDER, filename)

            try:
                # Download image
                img_response = requests.get(img_url, timeout=5)
                img = Image.open(BytesIO(img_response.content)).convert("RGB")

                # Resize image
                img = img.resize(TARGET_RESOLUTION)

                # Save image
                img.save(save_path)
                downloaded_count += 1
                print(f"✔ Saved: {save_path} (Resized to {TARGET_RESOLUTION})")

            except Exception as e:
                print(f"⚠ Error downloading {img_url}: {e}")

        page += 1  # Move to the next page

    print(f"\n✅ **Download complete** - {downloaded_count} images saved.")

# ✅ Run the downloader
download_images()
