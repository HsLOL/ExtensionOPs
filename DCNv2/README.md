## Compile cuda extension for DCNv2  
### original file tree  
```
|- DCNv2/  
   |-- dcn_v2.py  
   |-- testcpu.py  
   |-- testcuda.py  
   |-- __init__.py  
   |-- Makefile  
   |-- setup.py  
   |-- DCN/  
       |-- src/  
           |-- dcn_v2.h  
	   |-- vision.cpp  
	   |-- cpu/  
	       |-- dcn_v2_cpu.cpp  
	       |-- dcn_v2_im2col_cpu.cpp  
	       |-- dcn_v2_im2col_cpu.h  
	       |-- dcn_v2_psroi_pooling_cpu.cpp  
	       |-- vision.h  
	   |-- cuda/  
	       |-- dcn_v2_cuda.cu  
	       |-- dcn_v2_im2col_cuda.cu  
	       |-- dcn_v2_im2col_cuda.h  
	       |-- dcn_v2_psroi_pooling.cu  
	       |-- vision.h  
```
### Begin to compile cuda extension  
1. create conda environment  
```
conda create -n dcnv2 python=3.7 -y  
conda activate dcnv2  
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch  
```  
2. run make  
```
make  
```
### End compile  
### Result file tree  
```
|- DCNv2/
   |-- dcn_v2.py
   |-- testcpu.py
   |-- testcuda.py
   |-- __init__.py
   |-- Makefile
   |-- setup.py
   |-- _ext.cpython-37m-x86_64-linux-gnu.so (produced .so file)  
   |-- DCN/
       |-- src/
           |-- dcn_v2.h
           |-- vision.cpp
           |-- cpu/
               |-- dcn_v2_cpu.cpp
               |-- dcn_v2_im2col_cpu.cpp
               |-- dcn_v2_im2col_cpu.h
               |-- dcn_v2_psroi_pooling_cpu.cpp
               |-- vision.h
           |-- cuda/
               |-- dcn_v2_cuda.cu
               |-- dcn_v2_im2col_cuda.cu
               |-- dcn_v2_im2col_cuda.h
               |-- dcn_v2_psroi_pooling.cu
               |-- vision.h

```
### For gradcheck problem  
[Readme](https://github.com/CharlesShang/DCNv2)
