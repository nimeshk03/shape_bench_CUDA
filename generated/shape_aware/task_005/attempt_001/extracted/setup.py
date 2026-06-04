# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="rowwise_softmax_ext",
    ext_modules=[
        CUDAExtension(
            name="rowwise_softmax_ext",
            sources=["rowwise_softmax.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
