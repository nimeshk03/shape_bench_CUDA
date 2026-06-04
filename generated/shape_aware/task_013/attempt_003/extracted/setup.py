# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="task_013_ext",
    ext_modules=[
        CUDAExtension(
            name="task_013_ext",
            sources=["task_013_kernel.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
