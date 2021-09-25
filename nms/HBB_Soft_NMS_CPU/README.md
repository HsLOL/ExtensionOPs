## Compile cpp extension for horizontal bbox nms  
### original file tree:  
```
|-- HBB_NMS_CPU/  
    |-- cpu_nms.pyx  
    |-- __init__.py  
    |-- Makefile  
    |-- setup.py  
```
### Begin to compile cpp extension  
1. create conda environment  
```
conda create -n nms_cpu python=3.7 -y  
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch
```
2. If your conda environment don't have Cython, you should pip install Cython first  
```
pip install Cython
```
3. run make  
```
make
```
### End compile  
### Result file tree  
```
|-- HBB_NMS_CPU  
    |-- cpu_nms.pyx  
    |--__init__.py  
    |--Makefile  
    |--setup.py  
    |--`cpu_nms.cpp`  
    |--`cpu_soft_nms.cpython-37m-x86_64-linux-gnu.so`
```
