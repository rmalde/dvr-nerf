#install colmap and Ceres Solver

set -e #stop the script on error

sudo apt-get install \
    git \
    cmake \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libboost-test-dev \
    libeigen3-dev \
    libsuitesparse-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev

sudo apt-get install libcgal-qt5-dev


#Ceres
sudo apt-get install cmake
sudo apt-get install libgoogle-glog-dev libgflags-dev
sudo apt-get install libatlas-base-dev
sudo apt-get install libeigen3-dev
sudo apt-get install libsuitesparse-dev

if [ ! -f ceres-solver-2.1.0.tar.gz ]
then
  wget http://ceres-solver.org/ceres-solver-2.1.0.tar.gz
fi

if [ ! -f ceres-solver-2.1.0 ]
then
  tar zxf ceres-solver-2.1.0.tar.gz
fi
set +e
mkdir ceres-bin
set -e
cd ceres-bin
cmake ../ceres-solver-2.1.0
sudo make -j3
# sudo make test
sudo make install

cd ..
set +e
git clone https://github.com/colmap/colmap.git
set -e
cd colmap
git checkout dev
set +e
mkdir build
set -e
cd build
cmake ..
make -j
sudo make install
# # cleanup
# cd ..
# sudo rm -r ceres-bin
# rm ceres-solver-2.1.0
# rm ceres-solver-2.1.0.tar.gz
