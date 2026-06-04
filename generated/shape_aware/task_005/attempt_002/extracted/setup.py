# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="rowwise_softmax_cuda",
    ext_modules=[
        CUDAExtension(
            name="rowwise_softmax_cuda",
            sources=["rowwise_softmax.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-arch=sm_70",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
