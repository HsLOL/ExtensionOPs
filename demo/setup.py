from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='_CUDA',
    ext_modules=[
        CUDAExtension('sigmoid_cuda', [
            'src/sigmoid_cuda.cpp',
            'src/sigmoid_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
