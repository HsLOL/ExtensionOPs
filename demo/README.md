# A demo about compilation with C++ and CUDA for Python Extension.  
## Get Started  
A. Install requirements  
```
conda create -n env python=3.7
conda activate env
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch
```
B. Compile the Python Extension
```
make
```
C. Test the Python Extension
```
python test.py
```
## File Tree  
```
# Here is the project file tree.
demo/
    -src/
        -sigmoid_cuda.cpp
	-sigmoid_cuda_kernel.cu
    -makefile
    -setup.py
    -test.py
    -sigmoid_cuda.cpython-37m-x86_64-linux-gnu.so(generated *.so file)

```
