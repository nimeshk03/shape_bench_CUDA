# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="matmul_cuda",
    ext_modules=[
        CUDAExtension(
            name="matmul_cuda",
            sources=["matmul_cuda.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
