# setup.py  (build glue)
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="task010_ext",
    ext_modules=[
        CUDAExtension(
            name="task010_ext",
            sources=["task010_kernel.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
