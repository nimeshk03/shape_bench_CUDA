# setup.py / build glue
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="matrix_transpose_cuda",
    ext_modules=[
        CUDAExtension(
            name="matrix_transpose_cuda",
            sources=["matrix_transpose.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
