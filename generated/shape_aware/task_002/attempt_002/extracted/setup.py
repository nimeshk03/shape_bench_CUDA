# setup.py  (build glue)
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="rowwise_sum_ext",
    ext_modules=[
        CUDAExtension(
            name="rowwise_sum_ext",
            sources=["rowwise_sum.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
