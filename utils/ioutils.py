import os

MEDIA_SUBDIR = "media"
COLMAP_SUBDIR = "colmap" 
COLMAP_TEXT_SUBDIR = "text"
COLMAP_BIN_SUBDIR = "sparse/0" 
COLMAP_TRANSFORMS_FILENAME = "transforms.json" 

class DataDirs:
    def __init__(self, dataset_dir):
        self.media_dir = os.path.join(dataset_dir, MEDIA_SUBDIR)
        self.colmap_dir = os.path.join(dataset_dir, COLMAP_SUBDIR)
        self.colmap_text = os.path.join(self.colmap_dir, COLMAP_TEXT_SUBDIR)
        self.colmap_bin_dir = os.path.join(self.colmap_dir, COLMAP_BIN_SUBDIR)



