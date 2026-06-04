# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="batched_matmul_ext",
    ext_modules=[
        CUDAExtension(
            name="batched_matmul_ext",
            sources=["batched_matmul.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
