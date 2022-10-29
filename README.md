# dvr-nerf

## Setup

Install colmap and Ceres Solver. For Ubuntu, run:
```
cd scripts
./install_colmap.sh
```

For Windows, follow instructions at https://colmap.github.io/install.html

### Setup Issues
If you get error No CMAKE_CUDA_COMPILER could be found, run:
```
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64
export PATH=$PATH:$CUDA_HOME/bin
```
and then reload terminal. (Or try running and not reloading terminal)

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