import os
from glob import glob
from PIL import Image

def convert_images(root_dir):
    """
    Converts all .color.png images inside the root_dir (including subdirectories)
    to .color.jpg while keeping the same naming convention.
    """
    # Find all .color.png files in the directory and subdirectories
    png_images = glob(os.path.join(root_dir, "**", "*.color.png"), recursive=True)

    for png_path in png_images:
        # Generate new file name with .jpg extension
        jpg_path = png_path.replace(".color.png", ".color.jpg")

        try:
            # Open the PNG image
            with Image.open(png_path) as img:
                # Convert and save as JPEG
                img = img.convert("RGB")  # Ensures proper conversion
                img.save(jpg_path, "JPEG", quality=95)  # Save with high quality

            print(f"Converted: {png_path} -> {jpg_path}")

        except Exception as e:
            print(f"Failed to convert {png_path}: {e}")

if __name__ == "__main__":
    root_directory = '/root/dataset/tum_rgbd_color_rgb'
    if os.path.isdir(root_directory):
        convert_images(root_directory)
        print("Conversion completed.")
    else:
        print("Invalid directory. Please provide a valid path.")
