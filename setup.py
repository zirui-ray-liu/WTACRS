"""Install WTACRS."""
import setuptools
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def setup_package():
  long_description = "seq2seq"
  setuptools.setup(
      name='seq2seq',
      version='0.0.1',
      description='WTACRS',
      long_description=long_description,
      long_description_content_type='text/markdown',
      license='MIT License',
      packages=setuptools.find_packages(
          exclude=['docs', 'tests', 'scripts', 'examples']),
      dependency_links=[
          'https://download.pytorch.org/whl/torch_stable.html',
      ],
      classifiers=[
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.7.10',
      ],
      keywords='text nlp machinelearning',
      ext_modules=[
        CUDAExtension('seq2seq.backend.bit_relu_dropout',
            sources=[
            'seq2seq/backend/bit_relu_dropout/quantization.cpp',
            'seq2seq/backend/bit_relu_dropout/quantization_cuda_kernel.cu',
            ],
            extra_compile_args={'nvcc': ['--expt-extended-lambda']}
        ),
        CUDAExtension('seq2seq.projections.fwh_cuda',
            sources=[
            'seq2seq/projections/fwh_cuda/fwh_cpp.cpp',
            'seq2seq/projections/fwh_cuda/fwh_cu.cu',
            ]
        ),
        CUDAExtension('seq2seq.backend.inplace_layernorm',
            sources=[
            'seq2seq/backend/inplace_layernorm/inplace_layernorm.cpp',
            'seq2seq/backend/inplace_layernorm/inplace_layernorm_kernel.cu',
            ],
            extra_compile_args=['-O3']
        ),
      ],
      cmdclass={"build_ext": BuildExtension},
      install_requires=[
        'datasets==1.6.2',
        'scikit-learn==0.24.2',
        'tensorboard==2.5.0',
        'matplotlib==3.4.2',
        'transformers==4.6.0',
        'numpy==1.21.1'
      ],
  )


if __name__ == '__main__':
  setup_package()
