# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="task_016_ext",
    ext_modules=[
        CUDAExtension(
            name="task_016_ext",
            sources=["task_016_kernel.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
