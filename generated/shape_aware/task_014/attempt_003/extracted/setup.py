# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="task_014_ext",
    ext_modules=[
        CUDAExtension(
            name="task_014_ext",
            sources=["task_014_kernel.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
