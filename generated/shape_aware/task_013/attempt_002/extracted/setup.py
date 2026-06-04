# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='batched_transpose_cuda',
    ext_modules=[
        CUDAExtension(
            name='batched_transpose_cuda',
            sources=['batched_transpose.cu'],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
