# dvr-nerf


## Data
Put all of your data in `data` directory, for example `data/basketball/*.jpg` or `data/basketball/vid.mp4`


## Colmap 
For images run:
``` 
python scripts/colmap2nerf.py --run_colmap --images data/[folder_name]
```
For videos run:
```
python scripts/colmap2nerf.py --run_colmap --colmap_matcher sequential --video data/[folder_name]
```