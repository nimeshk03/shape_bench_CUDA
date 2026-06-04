# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="task015_ext",
    ext_modules=[
        CUDAExtension(
            name="task015_ext",
            sources=["task015_kernel.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
