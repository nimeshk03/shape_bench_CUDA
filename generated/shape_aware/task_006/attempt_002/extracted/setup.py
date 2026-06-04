# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="rowwise_layer_norm_ext",
    ext_modules=[
        CUDAExtension(
            name="rowwise_layer_norm_ext",
            sources=["rowwise_layer_norm.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
