# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="task_008_ext",
    ext_modules=[
        CUDAExtension(
            name="task_008_ext",
            sources=["task_008_kernel.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
