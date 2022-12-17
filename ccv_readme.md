# To run on the Brown University Cluster 
Run `bash scripts/load_env.sh`

Run `interact -n 2 -t 12:00:00 -m 128g -q gpu -X -g 2 -f geforce3090` - this gives you asccess to the GPUs which you actually need 

* `module load cuda/10.2`
* `module load cudnn/7.6.5`
* `module load gcc/8.2.0` 
* `module load anaconda/2020.02` 
`conda create -n $name_of_your_choice`
`source activate $name_of_your_choice`
 
Then proceed with installing the dependencies on https://github.com/yiwenchen1999/implicitObjDetection#dependencies


# Download the dataset on the cluster 
You can download the dataset by using the following command, but before you download check your quota first as the dataset is approximately 55GB - you have been warned!

`myquota` on the terminal will help you check what directory is free `~/scratch` should have a lot of space. 
 
`wget https://storage.googleapis.com/kubric-public/data/NeSFDatasets/NeSF%20datasets/toybox-13.tar.gz`

To unzip: 
`tar -xvf toybox-13.tar.gz`

# Running Replica Dataset on OSCAR - It "kind of" worked 
The biggest issue is actually running it on the right system and finding the right dependencies. I recommend Ubuntu. I tried with the Mac M1 (it won't work, trust me - the OpenGL is not compatible, you won't go very far). I tried Windows (it's a pain because all the systems require make cmake and are very Linux/Ubuntu dependent). 

So I recommend *Ubuntu* or Oscar, albeit I didn't actually make it with Oscar completely (Ubuntu probably works). 

By the way, this doesn't run with the gpu.. at least the viewer doesn't. I couldn't figure out how to let it actually display. 

## Replica and its Dependencies 
The Replica repository had "eigen" and "Pangolin" as the main dependencies under 'Replica-Dataset/3rdParty'
`git submodule update --init` in the Replica directory does pull the Pangolin and eigen, but I wouldn't recommend it. Just softlink it to a stable version. 

### Eigen 
On Oscar, I did:
* `module load eigen`
* Then I went to `Replica-Dataset/3rdParty` and did `ln -s eigen /users/$username/data/$username/csci2951/Replica-Dataset/3rdparty` 

## Pangolin 
This one is quite unwieldy and fragile. I just git cloned https://github.com/stevenlovegrove/Pangolin using the latest master branch at commit 5f78f502117b2ff9238ed63768fd859a8fa78ffd. Pangolin has **a lot of dependencies** that need to be resolved. If you can, just running `./scripts/install_prerequisites.sh -m brew/vcpkg/etc. all`  would be ideal, but Oscar didn't let me use my own package manger so the following is the workaround. 

Basically you have to download, at least all the recommended packages denoted in https://github.com/stevenlovegrove/Pangolin/blob/master/scripts/install_prerequisites.sh 

On Oscar:
* `module load opengl/nvidia-410.72 glew/1.13.0 gcc/10.2 python3 cmake/3.10.1 eigen python3 zstd libjpeg-turbo/2.0.2 ffmpeg `
* technically, you also have to load `libpng, tiff, lz4, Catch2` but I ignored it or installed it with conda or pip.. it was either preloaded or not needed for me, but check this other file for what conda env worked 
* `mkdir build`
* `cd build`
* `ccmake ..` - I recommend ccmake as it helps you configure other projects 
  - A main problem I had was that it couldn't find the glew library I had, so I had to change the `cmake/FindGLEW.make` file:
 - Under `IF (WIN32)`:
   - under ` FIND_PATH( GLEW_INCLUDE_DIR GL/glew.h` I added `/users/$username/anaconda/csci2951conda/include` 
   - under ` FIND_LIBRARY( GLEW_LIBRARY` I added `/gpfs/runtime/opt/glew/2.1.0/lib` which is where my glew resides
 - Under `ELSE (WIN32)`: 
    - under ` FIND_PATH( GLEW_INCLUDE_DIR GL/glew.h` I added `/users/$username/anaconda/csci2951conda/include` 
    - under `  FIND_LIBRARY( GLEW_LIBRARY` I added `/gpfs/runtime/opt/glew/2.1.0/lib64`
* CCMAKE Configurations type in `-t` to load more advanced options and update the following:
   Most of them are ppointing to the default `/usr/local` or something like that, but you have to directly tell the compiler info etc. or else you may get a `GNU` error: 
   - `CMAKE_CXX_COMPILER /gpfs/runtime/opt/gcc/10.2/bin/c++`
   - `CMAKE_CXX_COMPILER_AR /gpfs/runtime/opt/gcc/10.2/bin/gcc-ar `
   - `CMAKE_CXX_COMPILER_RANLIB /gpfs/runtime/opt/gcc/10.2/bin/gcc-ranlib`
   - `CMAKE_C_COMPILER /gpfs/runtime/opt/gcc/10.2/bin/gcc`
   - `CMAKE_C_COMPILER_AR /gpfs/runtime/opt/gcc/10.2/bin/gcc-ar`
   - `CMAKE_C_COMPILER_RANLIB /gpfs/runtime/opt/gcc/10.2/bin/gcc-ranlib` 
 * After you ccmake, then do `make -j`  
 * Then I went to `Replica-Dataset/3rdParty` and did `ln -s Pangolin $pathToPangolin` 
 
## Building Replica - Everything here is under the Replica-Dataset 
* `mkdir build`
* `cd build`
* `ccmake ..` 
* CCMAKE Configurations type in `-t` to load more advanced options and update the following:
   Most of them are ppointing to the default `/usr/local` or something like that, but you have to directly tell the compiler info etc. or else you may get a `GNU` error: 
   - `CMAKE_CXX_COMPILER /gpfs/runtime/opt/gcc/10.2/bin/c++`
   - `CMAKE_CXX_COMPILER_AR /gpfs/runtime/opt/gcc/10.2/bin/gcc-ar `
   - `CMAKE_CXX_COMPILER_RANLIB /gpfs/runtime/opt/gcc/10.2/bin/gcc-ranlib`
   - `CMAKE_C_COMPILER /gpfs/runtime/opt/gcc/10.2/bin/gcc`
   - `CMAKE_C_COMPILER_AR /gpfs/runtime/opt/gcc/10.2/bin/gcc-ar`
   - `CMAKE_C_COMPILER_RANLIB /gpfs/runtime/opt/gcc/10.2/bin/gcc-ranlib` 
* I had some `include` path issues (maybe they changed the paths who knows idk), so this is how I resolved them: 
   - Go to `ReplicaSDK/include/PTexLib.h` and replace `#include <pangolin/display/opengl_render_state.h>` with `#include <pangolin/gl/opengl_render_state.h>`
   - Go to ` ReplicaSDK/src/viewer.cpp` and replace `#include <pangolin/display/widgets/widgets.h>` with `#include <pangolin/display/widgets.h>` 

* `make -j` 

## Runnign Replica Dataset 
* `export  XDG_RUNTIME_DIR=""` --> idk the blank quotes make it work ;3
* `cd ~/Replica-Dataset/build/ReplicaSDK/`
* `./ReplicaViewer ~$dataPath/apartment_0/mesh.ply ~$dataPath/apartment_0/textures/ ~$dataPath/apartment_0/glass.sur`

# Display: 
This is what I ended up being able to render: 
![Screen Shot 2022-12-15 at 3 55 01 PM](https://user-images.githubusercontent.com/19434572/208214409-93d73496-d09e-4030-9c4b-f5ee0deef609.png)


# Running a jupyter notebook [needs update] 
Run `bash scripts/start_notebook.sh`
Take the printed ssh command and run it on a local terminal (this is not the oscar terminal but the one locally on your computer).
If the token does not show up it should in the whatever directory you ran the start_notebook script under `jupyter-log-xxxxxxx.txt` 


# Conda Env I made had the following packages: 
 libgcc_mutex             0.1                        main  
_openmp_mutex             5.1                       1_gnu  
blas                      1.0                         mkl  
blosc                     1.21.0               h4ff587b_1    anaconda
brotli                    1.0.9                h5eee18b_7    anaconda
brotli-bin                1.0.9                h5eee18b_7    anaconda
brunsli                   0.1                  h2531618_0    anaconda
bzip2                     1.0.8                h7b6447c_0  
c-ares                    1.18.1               h7f8727e_0    anaconda
ca-certificates           2022.10.11           h06a4308_0  
cairo                     1.16.0               h19f5f5c_2  
certifi                   2022.9.24       py310h06a4308_0  
cfitsio                   3.470                hf0d0db6_6    anaconda
charls                    2.2.0                h2531618_0    anaconda
charset-normalizer        2.1.1                    pypi_0    pypi
clip                      1.0                      pypi_0    pypi
cloudpickle               2.0.0              pyhd3eb1b0_0    anaconda
cytoolz                   0.11.0          py310h7f8727e_0    anaconda
dask-core                 2022.7.0        py310h06a4308_0    anaconda
eigen                     3.3.7                hd09550d_1  
expat                     2.4.9                h6a678d5_0   
fontconfig                2.14.1               h52c9d5c_1  
freetype                  2.12.1               h4a9f257_0  
fribidi                   1.0.10               h7b6447c_0  
fsspec                    2022.3.0        py310h06a4308_0    anaconda
ftfy                      6.1.1                    pypi_0    pypi
giflib                    5.2.1                h7b6447c_0  
glew                      2.1.0                h9c3ff4c_2    conda-forge
glib                      2.69.1               he621ea3_2  
graphite2                 1.3.14               h295c915_1  
harfbuzz                  4.3.0                hd55b92a_0  
icu                       58.2                 he6710b0_3  
idna                      3.4                      pypi_0    pypi
imagecodecs               2021.8.26       py310hecf7e94_1  
imageio                   2.19.3          py310h06a4308_0    anaconda
intel-openmp              2021.4.0          h06a4308_3561  
jpeg                      9e                   h7f8727e_0  
jxrlib                    1.1                  h7b6447c_2    anaconda
krb5                      1.19.2               hac12032_0    anaconda
lcms2                     2.12                 h3be6417_0  
ld_impl_linux-64          2.38                 h1181459_1  
lerc                      3.0                  h295c915_0  
libaec                    1.0.4                he6710b0_1    anaconda
libbrotlicommon           1.0.9                h5eee18b_7    anaconda
libbrotlidec              1.0.9                h5eee18b_7    anaconda
libbrotlienc              1.0.9                h5eee18b_7    anaconda
libcurl                   7.84.0               h91b91d3_0    anaconda
libdeflate                1.8                  h7f8727e_5  
libedit                   3.1.20210910         h7f8727e_0    anaconda
libev                     4.33                 h7f8727e_1    anaconda
libffi                    3.4.2                h6a678d5_6  
libgcc                    7.2.0                h69d50b8_2  
libgcc-ng                 11.2.0               h1234567_1  
libgfortran-ng            7.5.0               ha8ba4b0_17    anaconda
libgfortran4              7.5.0               ha8ba4b0_17    anaconda
libglu                    9.0.0                hf484d3e_1  
libgomp                   11.2.0               h1234567_1  
libnghttp2                1.46.0               hce63b2e_0    anaconda
libpng                    1.6.37               hbc83047_0  
libssh2                   1.10.0               h8f2d780_0    anaconda
libstdcxx-ng              11.2.0               h1234567_1  
libtiff                   4.4.0                hecacb30_2  
libuuid                   1.41.5               h5eee18b_0  
libwebp                   1.2.4                h11a3e52_0  
libwebp-base              1.2.4                h5eee18b_0  
libxcb                    1.15                 h7f8727e_0  
libxml2                   2.9.14               h74e7548_0  
libzopfli                 1.0.3                he6710b0_0    anaconda
locket                    1.0.0           py310h06a4308_0    anaconda
lz4-c                     1.9.3                h295c915_1  
mkl                       2021.4.0           h06a4308_640  
mkl-service               2.4.0           py310h7f8727e_0  
mkl_fft                   1.3.1           py310hd6ae3a3_0  
mkl_random                1.2.2           py310h00e6091_0  
ncurses                   6.3                  h5eee18b_3  
networkx                  2.8.4           py310h06a4308_0    anaconda
numpy                     1.23.4          py310hd5efca6_0  
numpy-base                1.23.4          py310h8e6c178_0  
nvidia-cublas-cu11        11.10.3.66               pypi_0    pypi
nvidia-cuda-nvrtc-cu11    11.7.99                  pypi_0    pypi
nvidia-cuda-runtime-cu11  11.7.99                  pypi_0    pypi
nvidia-cudnn-cu11         8.5.0.96                 pypi_0    pypi
openjpeg                  2.4.0                h3ad879b_0    anaconda
openssl                   1.1.1s               h7f8727e_0  
packaging                 21.3               pyhd3eb1b0_0    anaconda
pango                     1.50.7               h05da053_0  
partd                     1.2.0              pyhd3eb1b0_1    anaconda
pcre                      8.45                 h295c915_0  
pillow                    9.2.0           py310hace64e9_1    anaconda
pip                       22.2.2          py310h06a4308_0  
pixman                    0.40.0               h7f8727e_1  
pthread-stubs             0.4               h36c2ea0_1001    conda-forge
pyparsing                 3.0.4              pyhd3eb1b0_0    anaconda
python                    3.10.8               h7a1cb2a_1  
pywavelets                1.3.0           py310h7f8727e_0    anaconda
pyyaml                    6.0                      pypi_0    pypi
readline                  8.2                  h5eee18b_0  
regex                     2022.10.31               pypi_0    pypi
requests                  2.28.1                   pypi_0    pypi
scikit-image              0.19.2          py310h00e6091_0    anaconda
scipy                     1.7.3           py310hfa59a62_0    anaconda
setuptools                65.5.0          py310h06a4308_0  
six                       1.16.0             pyhd3eb1b0_1  
snappy                    1.1.9                h295c915_0    anaconda
sqlite                    3.40.0               h5082296_0  
tifffile                  2021.7.2           pyhd3eb1b0_2    anaconda
tk                        8.6.12               h1ccaba5_0  
toolz                     0.11.2             pyhd3eb1b0_0    anaconda
torch                     1.13.0                   pypi_0    pypi
torchvision               0.14.0                   pypi_0    pypi
tqdm                      4.64.1          py310h06a4308_0  
typing-extensions         4.4.0                    pypi_0    pypi
tzdata                    2022f                h04d1e81_0  
urllib3                   1.26.13                  pypi_0    pypi
wcwidth                   0.2.5                    pypi_0    pypi
wheel                     0.37.1             pyhd3eb1b0_0  
xorg-kbproto              1.0.7             h7f98852_1002    conda-forge
xorg-libx11               1.7.2                h7f98852_0    conda-forge
xorg-libxau               1.0.9                h7f98852_0    conda-forge
xorg-libxdmcp             1.1.3                h7f98852_0    conda-forge
xorg-libxext              1.3.4                h7f98852_1    conda-forge
xorg-xextproto            7.3.0             h7f98852_1002    conda-forge
xorg-xproto               7.0.31            h7f98852_1007    conda-forge
xz                        5.2.6                h5eee18b_0  
yaml                      0.2.5                h7b6447c_0    anaconda
zfp                       0.5.5                h295c915_6    anaconda
zlib                      1.2.13               h5eee18b_0  
zstd                      1.5.2                ha4553b6_0  


