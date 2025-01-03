import glob
import os
from setuptools import find_packages
from setuptools import setup
import torch
from torch.utils.cpp_extension import CUDA_HOME
from torch.utils.cpp_extension import CppExtension
from torch.utils.cpp_extension import CUDAExtension


def get_extensions():

    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "vinvl", "maskrcnn_benchmark", "csrc")

    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))

    sources = main_file + source_cpu
    extension = CppExtension

    extra_compile_args = {"cxx": []}
    define_macros = []

    if (torch.cuda.is_available() and CUDA_HOME is not None) or os.getenv("FORCE_CUDA", "0") == "1":
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "vinvl.maskrcnn_benchmark._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


setup(
    name="vinvl",
    version="0.1.0",
    author="Mahmood Anaam",
    url="https://github.com/Mahmood-Anaam/vinvl.git",
    description="object detection in pytorch",
    packages=find_packages(exclude=("configs", "tests","notebooks","assets","scripts")),
    install_requires = [
    "anytree==2.12.1",
    "cityscapesScripts==2.2.4",
    "clint==0.5.1",
    "Cython==3.0.11",
    "einops==0.8.0",
    "huggingface-hub==0.27.0",
    "ninja==1.11.1.3",
    "opencv-python==4.10.0.84",
    "pillow==11.1.0",
    "PyYAML==6.0.2",
    "timm==1.0.12",
    "torch==2.5.1",
    "torchvision==0.20.1",
    "tqdm==4.67.1",
    "transformers==4.47.1",
    "twine==6.0.1",
    "urllib3==2.3.0",
    "yacs==0.1.8",
    ],

    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},

    package_data={
        "vinvl": [
            os.path.join("sgg_configs","vgattr","*.yaml"),
            os.path.join("pretrained_model","vinvl_vg_x152c4","*"),
        ]
    },

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
