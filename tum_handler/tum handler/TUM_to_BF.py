import os
import numpy as np
from scipy.spatial.transform import Rotation as R
import argparse

# Step 1: Match and rename RGB/Depth/GT files
def combine_and_rename_files(tum_folder, output_folder, margin=0.02):
    """
    Matches RGB, depth, and ground truth files based on timestamps within a margin, renames them to BundleFusion format, 
    and saves to the output folder.
    """
    pose_file = os.path.join(tum_folder, "groundtruth.txt")

    # Create output folders
    os.makedirs(output_folder, exist_ok=True)

    # Dynamically list files in rgb and depth directories
    rgb_path = os.path.join(tum_folder, "rgb")
    depth_path = os.path.join(tum_folder, "depth")

    rgb_files = sorted(os.listdir(rgb_path))
    depth_files = sorted(os.listdir(depth_path))

    # Extract timestamps from filenames
    rgb_entries = [(float(f[:-4]), f) for f in rgb_files if f.endswith(".png") or f.endswith(".jpg")]
    depth_entries = [(float(f[:-4]), f) for f in depth_files if f.endswith(".png")]

    # Load poses
    pose_file = os.path.join(tum_folder, "groundtruth.txt")
    with open(pose_file, "r") as f:
        pose_lines = [line.strip() for line in f if not line.startswith("#")]
    pose_entries = [(float(line.split()[0]), line.split()[1:]) for line in pose_lines]


    frame_counter = 0
    for _, (rgb_ts, rgb_filename) in enumerate(rgb_entries):
        frame_id = f"frame-{frame_counter:06d}"
        
        # Find closest depth
        closest_depth = min(depth_entries, key=lambda x: abs(rgb_ts - x[0]))
        if abs(rgb_ts - closest_depth[0]) > margin:
            print("skipped")
            continue

        # Find closest pose
        closest_pose = min(pose_entries, key=lambda x: abs(rgb_ts - x[0]))
        if abs(rgb_ts - closest_pose[0]) > margin:
            print("skipped")
            continue
        depth_entries.remove(closest_depth)
        pose_entries.remove(closest_pose)
        frame_counter += 1

        # RGB
        rgb_src = os.path.join(rgb_path, rgb_filename)
        rgb_dst = os.path.join(output_folder, f"{frame_id}.color.png")
        os.rename(rgb_src, rgb_dst)

        # Depth
        depth_src = os.path.join(depth_path, closest_depth[1])
        print("here" ,depth_src)
        depth_dst = os.path.join(output_folder, f"{frame_id}.depth.png")
        os.rename(depth_src, depth_dst)

        # Pose
        tx, ty, tz, qx, qy, qz, qw = map(float, closest_pose[1])
        rotation = R.from_quat([qx, qy, qz, qw]).as_matrix()
        pose_matrix = np.eye(4)
        pose_matrix[:3, :3] = rotation
        pose_matrix[:3, 3] = [tx, ty, tz]

        pose_dst = os.path.join(output_folder, f"{frame_id}.pose.txt")
        np.savetxt(pose_dst, pose_matrix, fmt="%.6f")

    print(f"Files matched, renamed, and saved in {output_folder}")
# Step 2: Generate info.txt based on the prefix
# Step 2: Generate info.txt based on the prefix
def generate_info_txt(output_folder, folder_name):
    """
    Generates the info.txt file with intrinsics based on the dataset prefix (freiburg1, freiburg2, freiburg3).
    """
    intrinsics = {
        "freiburg1": "517.3 0 318.6 0 0 516.5 255.3 0 0 0 1 0 0 0 0 1",
        "freiburg2": "520.9 0 325.1 0 0 521.0 249.7 0 0 0 1 0 0 0 0 1",
        "freiburg3": "535.4 0 320.1 0 0 539.2 247.6 0 0 0 1 0 0 0 0 1"
    }

    # Default values
    color_intrinsic = "525.0 0 319.5 0 0 525.0 239.5 0 0 0 1 0 0 0 0 1"
    depth_intrinsic = "525.0 0 319.5 0 0 525.0 239.5 0 0 0 1 0 0 0 0 1"

    # Update intrinsics based on dataset
    if "freiburg1" in folder_name.lower():
        color_intrinsic = intrinsics["freiburg1"]
        depth_intrinsic = intrinsics["freiburg1"]
    elif "freiburg2" in folder_name.lower():
        color_intrinsic = intrinsics["freiburg2"]
        depth_intrinsic = intrinsics["freiburg2"]
    elif "freiburg3" in folder_name.lower():
        color_intrinsic = intrinsics["freiburg3"]
        depth_intrinsic = intrinsics["freiburg3"]

    info_path = os.path.join(output_folder, "info.txt")
    with open(info_path, "w") as f:
        f.write("m_versionNumber = 4\n")
        f.write("m_sensorName = Kinect\n")
        f.write("m_colorWidth = 640\n")
        f.write("m_colorHeight = 480\n")
        f.write("m_depthWidth = 640\n")
        f.write("m_depthHeight = 480\n")
        f.write("m_depthShift = 5000\n")
        f.write(f"m_calibrationColorIntrinsic = {color_intrinsic}\n")
        f.write("m_calibrationColorExtrinsic = 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1\n")
        f.write(f"m_calibrationDepthIntrinsic = {depth_intrinsic}\n")
        f.write("m_calibrationDepthExtrinsic = 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1\n")

    print(f"info.txt generated in {output_folder}")

# Step 3: Transform poses and save in BundleFusion format
def transform_poses(tum_folder, output_folder):
    """
    Transforms TUM poses to 4x4 transformation matrices and saves them in BundleFusion format.
    """
    pose_file = os.path.join(tum_folder, "groundtruth.txt")
    pose_output_folder = output_folder
    os.makedirs(pose_output_folder, exist_ok=True)

    with open(pose_file, "r") as f:
        lines = f.readlines()

    for line in lines:
        if line.startswith("#") or line.strip() == "":
            continue

        # Parse timestamp and pose
        data = line.strip().split()
        timestamp = data[0].replace(".", "")  # Format the timestamp
        tx, ty, tz = map(float, data[1:4])
        qx, qy, qz, qw = map(float, data[4:])

        # Convert quaternion to rotation matrix
        rotation = R.from_quat([qx, qy, qz, qw]).as_matrix()

        # Construct 4x4 transformation matrix
        pose_matrix = np.eye(4)
        pose_matrix[:3, :3] = rotation
        pose_matrix[:3, 3] = [tx, ty, tz]

        # Save the pose matrix
        pose_path = os.path.join(pose_output_folder, f"frame-{timestamp}.pose.txt")
        np.savetxt(pose_path, pose_matrix, fmt="%.6f")

    print(f"Poses transformed and saved in {pose_output_folder}")

# Main script
def main(tum_folder):
    # Determine prefix (f1, f2, or f3) from the folder name
    folder_name = os.path.basename(tum_folder)
  

    # Perform the tasks
    combine_and_rename_files(tum_folder, tum_folder)
    generate_info_txt(tum_folder,folder_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert TUM RGB-D dataset to BundleFusion format.")
    parser.add_argument("tum_folder", type=str, help="Path to the TUM RGB-D dataset folder.")

    args = parser.parse_args()
    main(args.tum_folder)
