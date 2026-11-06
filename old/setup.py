# setup.py
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import sys

# Define os argumentos de compilação para paralelismo (OpenMP)
# Varia de acordo com o sistema operacional
extra_compile_args = []
extra_link_args = []
if sys.platform == 'win32':
    # Para o compilador da Microsoft (MSVC)
    extra_compile_args = ['/openmp']
    extra_link_args = ['/openmp']
else:
    # Para GCC/Clang em Linux/macOS
    extra_compile_args = ['-fopenmp']
    extra_link_args = ['-fopenmp']


# Define o módulo a ser compilado
extensions = [
    Extension(
        "keyer_optimizer",
        ["keyer_optimizer.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args
    )
]

setup(
    ext_modules=cythonize(extensions),
    include_dirs=[numpy.get_include()]
)
