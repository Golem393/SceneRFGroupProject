#!/usr/bin/env python3

import os
import sys
import numpy as np
import trimesh
import pyrender
import imageio
from PIL import Image

def main():
    if len(sys.argv) < 2:
        print("Usage: python replica_process.py /path/to/replica_scene")
        sys.exit(1)
    
    # ------------------------------------------------------------
    # 1) Parse input and set up output folder
    # ------------------------------------------------------------
    scene_path = os.path.abspath(sys.argv[1])
    if not os.path.isdir(scene_path):
        print(f"Error: '{scene_path}' is not a valid directory.")
        sys.exit(1)
    
    # e.g., if scene_path = /path/to/replica_scene
    # then parent_dir = /path/to
    parent_dir = os.path.dirname(scene_path)
    scene_folder_name = os.path.basename(scene_path)
    
    # We'll create .../replica_proc/replica_scene
    out_parent_dir = os.path.join(parent_dir, "replica_proc")
    output_dir = os.path.join(out_parent_dir, scene_folder_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # ------------------------------------------------------------
    # 2) Write the info.txt with the dummy intrinsics/extrinsics
    # ------------------------------------------------------------
    # (These values are the ones you provided as an example.)
    info_txt_content = """m_versionNumber = 4
m_sensorName = Kinect
m_colorWidth = 640
m_colorHeight = 480
m_depthWidth = 640
m_depthHeight = 480
m_depthShift = 5000
m_calibrationColorIntrinsic = 517.3 0 318.6 0 0 516.5 255.3 0 0 0 1 0 0 0 0 1
m_calibrationColorExtrinsic = 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1
m_calibrationDepthIntrinsic = 517.3 0 318.6 0 0 516.5 255.3 0 0 0 1 0 0 0 0 1
m_calibrationDepthExtrinsic = 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1
"""
    info_txt_path = os.path.join(output_dir, "info.txt")
    with open(info_txt_path, 'w') as f:
        f.write(info_txt_content)
    print(f"Written intrinsics info to {info_txt_path}")
    
    # ------------------------------------------------------------
    # 3) Load the mesh (assuming it references all .hdr textures)
    # ------------------------------------------------------------
    mesh_ply_path = os.path.join(scene_path, "mesh.ply")
    if not os.path.exists(mesh_ply_path):
        print(f"Error: Could not find '{mesh_ply_path}'")
        sys.exit(1)
    
    print(f"Loading mesh from: {mesh_ply_path}")
    mesh = trimesh.load(mesh_ply_path)
    
    # Convert to a pyrender-compatible mesh
    # (If mesh has multiple materials, this returns a list of sub-meshes;
    #  pyrender.Mesh.from_trimesh can also handle that.)
    pyr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    
    # Create a scene in pyrender
    scene = pyrender.Scene()
    scene.add(pyr_mesh)
    
    # ------------------------------------------------------------
    # 4) Set up camera intrinsics
    # ------------------------------------------------------------
    # The dummy intrinsics from info.txt are:
    #   fx = 517.3, fy = 516.5, cx = 318.6, cy = 255.3
    #   image size 640x480
    # We can create an IntrinsicsCamera in pyrender with these values:
    fx = 517.3
    fy = 516.5
    cx = 318.6
    cy = 255.3
    width = 640
    height = 480
    
    camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, 
                                       znear=0.01, zfar=1000.0,
                                       width=width, height=height)
    camera_pose = np.eye(4)  # Will update per frame
    camera_node = scene.add(camera, pose=camera_pose)
    
    # ------------------------------------------------------------
    # 5) Create an offscreen renderer
    # ------------------------------------------------------------
    r = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    
    # ------------------------------------------------------------
    # 6) Define a simple camera trajectory
    # ------------------------------------------------------------
    num_frames = 30
    radius = 5.0   # Radius of circular path
    cam_height = 1.5
    angles = np.linspace(0, 2*np.pi, num_frames, endpoint=False)
    
    # ------------------------------------------------------------
    # 7) Render frames
    # ------------------------------------------------------------
    for i, angle in enumerate(angles):
        # Camera position on a circle
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = cam_height
        
        # Simple rotation so the camera looks toward the origin
        # We'll rotate around the Y-axis, such that the camera
        # faces the center (0,0,0) from (x,y,z).
        look_dir = np.array([0,0,0]) - np.array([x,y,z])
        look_dir /= np.linalg.norm(look_dir)
        
        # A quick way to create a look-at matrix is to define
        # forward, right, and up vectors. We'll do it manually:
        forward = look_dir
        # approximate up vector
        up = np.array([0, 1, 0], dtype=np.float32)
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        
        # Build rotation (3x3)
        rot = np.eye(3)
        rot[0, :] = right
        rot[1, :] = up
        rot[2, :] = -forward  # pyrender's camera looks -Z in its local frame
        
        camera_pose = np.eye(4)
        camera_pose[:3, :3] = rot
        camera_pose[:3, 3] = [x, y, z]
        
        # Update the pose in the scene
        scene.set_pose(camera_node, pose=camera_pose)
        
        # Render color and depth
        color, depth = r.render(scene)
        
        # Save RGB image
        rgb_path = os.path.join(output_dir, f"frame-{i:03d}.rgb.png")
        imageio.imwrite(rgb_path, color)
        
        # Save depth image (in 16-bit PNG, scaled by 1000 => mm)
        depth_path = os.path.join(output_dir, f"frame-{i:03d}.depth.png")
        depth_scaled = (depth * 1000).astype(np.uint16)
        imageio.imwrite(depth_path, depth_scaled)
        
        # Save camera pose
        pose_path = os.path.join(output_dir, f"frame-{i:03d}.pose.txt")
        np.savetxt(pose_path, camera_pose)
        
        print(f"Saved frame {i:03d}")
    
    # ------------------------------------------------------------
    # 8) Clean up
    # ------------------------------------------------------------
    r.delete()
    print(f"All done. Output frames and info.txt are in: {output_dir}")

if __name__ == "__main__":
    main()
