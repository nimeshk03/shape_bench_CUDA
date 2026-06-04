# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="task_006_ext",
    ext_modules=[
        CUDAExtension(
            name="task_006_ext",
            sources=["task_006_kernel.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
