from distutils.core import setup
from Cython.Build import cythonize
setup(ext_modules = cythonize(["config.py",
                               "functional.py",
                               "layers.py",
                               "trainer.py",
                               "dataloader.py",
                               "approx_utils.py",
                               "glue_trainer.py",
                               "scheme.py",
                               ]))