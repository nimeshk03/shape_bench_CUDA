# setup.py / build glue
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="layer_norm_cuda",
    ext_modules=[
        CUDAExtension(
            name="layer_norm_cuda",
            sources=["layer_norm_cuda.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
