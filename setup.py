import os

from setuptools import  setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from distutils.sysconfig import get_python_inc

python_include_dir = get_python_inc()

def make_cuda_ext(name, module, sources):
    cuda_ext = CUDAExtension(
        name='%s.%s' % (module, name),
        sources=[os.path.join(*module.split('.'), src) for src in sources]
    )
    return cuda_ext

'''
debug options: extra_compile_args={'cxx': ['-std=c++17 -O3 -g'],
            'nvcc':  [ "-std=c++17", "--gpu-architecture=sm_80", "-g", "-G",]},
'''
if __name__ == '__main__':

    setup(
    name='generateGSneural',
    ext_modules=[
        CUDAExtension('fusedKernels', 
        sources=  ['generateGSneural.cu',],
        include_dirs=[python_include_dir,
                # os.path.join(os.getcwd(),"cuda_kernel/include"),
                ],

        extra_compile_args={
            "cxx": ["-std=c++17"],
            "nvcc": [
                "-O3",
                "-use_fast_math",
                "-std=c++17",
                # "--gpu-architecture=sm_80",
            ],
        },
        ),
        CUDAExtension('FusedMatmul', 
            sources = ['fusedMatmul.cu',],
            extra_compile_args={
            "cxx": [ 
                # "-g","-G",
                    "-std=c++17"
                   ],
            "nvcc": [
                "-O3",
                # "-G",
                # "-g",
                # "-lineinfo",
                "-use_fast_math",
                "-std=c++17",
                # "--gpu-architecture=sm_80",
            ],},
            # extra_link_args=["-o./complex2int8_vec"],
        ),

    ],
    cmdclass={
        'build_ext': BuildExtension
        }
    )

   
