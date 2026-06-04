# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="task_007_ext",
    ext_modules=[
        CUDAExtension(
            name="task_007_ext",
            sources=["task_007_ext.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
