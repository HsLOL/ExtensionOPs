## Compile CUDA extension for horizontal bbox nms  
### original file tree:  
```
|-- HBB_NMS_GPU/
    |-- nms/
        |-- gpu_nms.pyx  
        |-- __init__.py
        |-- nms_kernel.cu
        |-- gpu_nms.hpp
    |-- Makefile  
    |-- setup.py
    |-- README.md
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
**Note**  
```
If you meet the error like this when you run `make`
Traceback (most recent call last):
  File "setup.py", line 58, in <module>
    CUDA = locate_cuda()
  File "setup.py", line 55, in locate_cuda
    raise EnvironmentError('The CUDA %s path could not be located in %s' % (k, v))
OSError: The CUDA lib64 path could not be located in /usr/lib64
Makefile:2: recipe for target 'all' failed
make: *** [all] Error 1

You should modify the 'lib64' to 'lib' in setup.py file
```
### End compile  
### Result file tree  
```
|-- HBB_NMS_GPU/
    |-- nms/
        |-- gpu_nms.pyx  
        |-- __init__.py
	|-- nms_kernel.cu
	|-- gpu_nms.hpp
	|-- `gpu_nms.cpp`
	|-- `gpu_nms.cpython-37m-x86_64-linux-gnu.so`
    |-- Makefile  
    |-- setup.py
    |-- README.md
```
