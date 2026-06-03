# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="elementwise_add_relu_cuda",
    ext_modules=[
        CUDAExtension(
            name="elementwise_add_relu_cuda",
            sources=["elementwise_add_relu.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
