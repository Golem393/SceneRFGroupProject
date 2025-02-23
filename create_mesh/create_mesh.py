import open3d as o3d
import os
import pickle
import numpy as np

relative_path = "/root/SceneRFGroupProject/evaluation/bf/original/recon/tsdf/copyroom"
abs_path = os.path.join(os.getcwd(), relative_path)
all_pkl_files = os.listdir(abs_path)
for pkl_file in all_pkl_files:
    with open(os.path.join(relative_path, pkl_file), 'rb') as f:
        data = pickle.load(f)
        mesh = o3d.geometry.TriangleMesh()
        mesh.triangle_normals = o3d.utility.Vector3dVector(data['norms'])
        mesh.vertices = o3d.utility.Vector3dVector(data['verts'])
        mesh.triangles = o3d.utility.Vector3iVector(data['faces'])
        mesh.vertex_colors = o3d.utility.Vector3dVector(data['colors'].astype(float) / 255.0)

        o3d.visualization.draw_geometries([mesh])
        #save ad png