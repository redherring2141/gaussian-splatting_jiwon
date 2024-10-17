#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#


from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))


'''
class get_pybind_include(object):
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """
    def __str__(self):
        import pybind11
        return pybind11.get_include()

ext_modules = [
    Extension(
        'your_module_name',
        ['source1.cpp', 'source2.cu'],
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            get_pybind_include(user=True)
        ],
        language='c++',
        extra_compile_args=['-O0', '-g', '--generate-code=arch=compute_52,code=sm_52'],
    ),
]

setup(
    name='your_module_name',
    version='0.0.1',
    author='Your Name',
    author_email='yourname@example.com',
    description='A description of your project',
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.5.0'],
    cmdclass={'build_ext': build_ext},
    zip_safe=False,
)
'''


setup(
    name="diff_gaussian_rasterization",
    packages=['diff_gaussian_rasterization'],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization._C",
            sources=[
            "cuda_rasterizer/rasterizer_impl.cu",
            "cuda_rasterizer/forward.cu",
            "cuda_rasterizer/backward.cu",
            "rasterize_points.cu",
            "ext.cpp"],
            extra_compile_args={"nvcc": ["-O0", "-Xcompiler", "-fPIC", "-G", "-g", 
                                         "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")],
                                'cxx': ["-g"]},
            extra_link_args=["-shared"]
            )
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)



'''
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

cxx_compiler_flags = ['-g', '-O0']

print("setting up...")

setup(
    name="diff_gaussian_rasterization",
    packages=['diff_gaussian_rasterization'],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization._C",
            sources=[
            "cuda_rasterizer/rasterizer_impl.cu",
            "cuda_rasterizer/forward.cu",
            "cuda_rasterizer/backward.cu",
            "rasterize_points.cu",
            "ext.cpp"],
            #extra_compile_args={"nvcc": ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]})
            extra_compile_args={"nvcc": ["-g", "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")], "cxx": cxx_compiler_flags})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
'''