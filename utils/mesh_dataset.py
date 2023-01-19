from torch.utils.data import Dataset
import trimesh
import numpy as np
from plyfile import PlyElement, PlyData
"""
This file is a util script to convert a mesh file into points for periodic function
"""

# Only for exporting the pointcloud
def export_pointcloud(vertices, out_file):
    assert(vertices.shape[1] == 3)
    vertices = vertices.astype(np.float32)
    vector_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    vertices = vertices.view(dtype=vector_dtype).flatten()
    plyel = PlyElement.describe(vertices, 'vertex')
    plydata = PlyData([plyel], text=True)
    plydata.write(out_file)


class MeshDataset(Dataset):
    def __init__(self, mesh_file_path='workspace/snorlax.stl'):
        trimesh.load(mesh_file_path)



# def as_mesh(scene_or_mesh):
#     """
#     Convert a possible scene to a mesh.

#     If conversion occurs, the returned mesh has only vertex and face data.
#     """
#     if isinstance(scene_or_mesh, trimesh.Scene):
#         if len(scene_or_mesh.geometry) == 0:
#             mesh = None  # empty scene
#         else:
#             # we lose texture information here
#             mesh = trimesh.util.concatenate(
#                 tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
#                     for g in scene_or_mesh.geometry.values()))
#     else:
#         assert(isinstance(mesh, trimesh.Trimesh))
#         mesh = scene_or_mesh
#     return mesh

mesh = trimesh.load('workspace/snorlax.stl', use_ambree=False)
# mesh = as_mesh(mesh)

print("Watertight: ", mesh.is_watertight)
# quit()
assert trimesh.repair.fill_holes(mesh)

print("Bounds")
print(mesh.bounds)
lower_bounds = mesh.bounds[0]
upper_bounds = mesh.bounds[1]
rand = np.random.rand(10_000, 3)
points = lower_bounds + (upper_bounds-lower_bounds) * rand

# print(mesh.geometry)

# points = 100 * np.random.rand(100000, 3)
# points, face_index = trimesh.sample.sample_surface(mesh, 10_000)
# print(points)
contains = mesh.contains(points)
print(f"Total Contains: {sum(contains)}")
print(f"Percent contains: {sum(contains)/points.shape[0]}")
# export_pointcloud(points[contains], 'workspace/snorlax_points.ply')

np.save('data/snorlax/contains.npy', contains)
np.save('data/snorlax/points.npy', points)

# mesh.export('workspace/bb8_watertight.obj')
# print("Watertight: ", mesh.is_watertight)