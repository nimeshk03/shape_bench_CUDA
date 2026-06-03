# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="add_relu_cuda",
    ext_modules=[
        CUDAExtension(
            name="add_relu_cuda",
            sources=["add_relu_cuda.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-arch=sm_80",
                    "--ptxas-options=-v",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
