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
    "appdirs==1.4.4",
    "args==0.1.0",
    "backports.tarfile==1.2.0",
    "certifi==2024.12.14",
    "charset-normalizer==3.4.1",
    "cityscapesScripts==2.2.4",
    "clint==0.5.1",
    "colorama==0.4.6",
    "coloredlogs==15.0.1",
    "contourpy==1.3.1",
    "cycler==0.12.1",
    "Cython==3.0.11",
    "docutils==0.21.2",
    "einops==0.8.0",
    "filelock==3.16.1",
    "fonttools==4.55.3",
    "fsspec==2024.12.0",
    "huggingface-hub==0.27.0",
    "humanfriendly==10.0",
    "idna==3.10",
    "importlib_metadata==8.5.0",
    "jaraco.classes==3.4.0",
    "jaraco.context==6.0.1",
    "jaraco.functools==4.1.0",
    "Jinja2==3.1.5",
    "keyring==25.6.0",
    "kiwisolver==1.4.8",
    "markdown-it-py==3.0.0",
    "MarkupSafe==3.0.2",
    "matplotlib==3.10.0",
    "mdurl==0.1.2",
    "more-itertools==10.5.0",
    "mpmath==1.3.0",
    "networkx==3.4.2",
    "nh3==0.2.20",
    "ninja==1.11.1.3",
    "numpy==2.2.1",
    "opencv-python==4.10.0.84",
    "packaging==24.2",
    "pillow==11.1.0",
    "pkginfo==1.12.0",
    "Pygments==2.18.0",
    "pyparsing==3.2.1",
    "pyquaternion==0.9.9",
    "pyreadline3==3.5.4",
    "python-dateutil==2.9.0.post0",
    "pywin32-ctypes==0.2.3",
    "PyYAML==6.0.2",
    "readme_renderer==44.0",
    "regex==2024.11.6",
    "requests==2.32.3",
    "requests-toolbelt==1.0.0",
    "rfc3986==2.0.0",
    "rich==13.9.4",
    "safetensors==0.5.0",
    "six==1.17.0",
    "sympy==1.13.1",
    "timm==1.0.12",
    "tokenizers==0.21.0",
    "torch==2.5.1",
    "torchvision==0.20.1",
    "tqdm==4.67.1",
    "transformers==4.47.1",
    "twine==6.0.1",
    "typing==3.7.4.3",
    "typing_extensions==4.12.2",
    "urllib3==2.3.0",
    "yacs==0.1.8",
    "zipp==3.21.0"
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
