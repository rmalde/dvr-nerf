from os.path import join

MEDIA_SUBDIR = "media"
COLMAP_SUBDIR = "colmap" 
COLMAP_TEXT_SUBDIR = "text"
COLMAP_BIN_SUBDIR = "sparse/0" 
WORKSPACE_SUBDIR = "workspace" 
COLMAP_TRANSFORMS_FILENAME = "transforms.json" 
MESH_SUBDIR = "mesh"

class DataDirs:
    def __init__(self, dataset_dir):
        self.media_dir = join(dataset_dir, MEDIA_SUBDIR)
        self.colmap_dir = join(dataset_dir, COLMAP_SUBDIR)
        self.colmap_text = join(self.colmap_dir, COLMAP_TEXT_SUBDIR)
        self.colmap_bin_dir = join(self.colmap_dir, COLMAP_BIN_SUBDIR)
        self.workspace = join(dataset_dir, WORKSPACE_SUBDIR)
        self.mesh_dir = join(dataset_dir, MESH_SUBDIR)

