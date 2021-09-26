## Compile CUDA extension for oriented bbox nms  
### original file tree:  
```
|-- OBB_NMS_GPU/
    |-- src/
        |-- rotate_polygon_nms.cpp  
        |-- rotate_polygon_nms_kernel.cu  
    |-- build.sh  
    |-- __init__.py
    |-- setup.py
    |-- README.md
```
### Begin to compile CUDA extension  
1. create conda environment  
```
conda create -n nms_cpu python=3.7 -y  
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch
```
2. run bash build.sh
```
bash build.sh
```
### End compile  
### Result file tree  
```
|-- OBB_NMS_GPU/
    |-- src/
        |-- rotate_polygon_nms.cpp  
        |-- rotate_polygon_nms_kernel.cu  
    |-- build/
    |-- build.sh
    |-- __init__.py
    |-- setup.py
    |-- README.md
    |-- `r_nms.cpython-37m-x86_64-linux-gnu.so`
```

