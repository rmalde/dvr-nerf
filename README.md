# DVR-NeRF: Dynamic NeRFs in VR
Independent research on optimizing mesh construction of Neural Radiance Fields (NeRFs) for real-time rendering in Virtual Reality

## Setup

Install colmap with `sudo apt install colmap`

## Data
Put all of your data in `data` directory, structured as follows `data/[dataset_name]/media/*.jpg` or `data/basketball/vid.mp4`


## Colmap 
For images run:
``` 
python scripts/colmap2nerf.py --run_colmap --dir data/[folder_name]
```
For videos run:
```
python scripts/colmap2nerf.py --run_colmap --from_video --colmap_matcher sequential --video_fps 5 --dir data/[folder_name]
```

## Train NeRF
Run
```
python3 train.py --model_type tensorf --data_dir data/[dataset_name]
```
