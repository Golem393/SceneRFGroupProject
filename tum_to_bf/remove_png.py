import os

def remove_color_png_files(directory):
    """Recursively remove all files ending with '_color.png' in the given directory."""
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('color.png'):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")

if __name__ == "__main__":
    dir_path = '/root/dataset/tum_rgbd_color_rgb'
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        remove_color_png_files(dir_path)
    else:
        print("Invalid directory path!")
