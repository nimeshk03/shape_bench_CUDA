# setup.py / build glue
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="task_012_ext",
    ext_modules=[
        CUDAExtension(
            name="task_012_ext",
            sources=["task_012_kernel.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
