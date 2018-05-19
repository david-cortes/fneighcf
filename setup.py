from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy
import platform

if platform.system() == "Windows":
	ext_mod = Extension("fneighcf.cython_loops",
                             sources=["fneighcf/cython_loops.pyx"],
                             include_dirs=[numpy.get_include()],
                             extra_compile_args=['/openmp'])
else:
	ext_mod = Extension("fneighcf.cython_loops",
                             sources=["fneighcf/cython_loops.pyx"],
                             include_dirs=[numpy.get_include()],
                             extra_compile_args=['-Ofast','-fopenmp'],
                             extra_link_args=['-fopenmp'])


setup(
  name = 'fneighcf',
  packages = ['fneighcf'],
  install_requires=[
   'pandas>=0.18.0',
   'numpy',
   'scipy',
   'cython'
],
  version = '0.2.2',
  description = 'Recommender system based on parameterized Item-Item effects',
  author = 'David Cortes',
  author_email = 'david.cortes.rivera@gmail.com',
  url = 'https://github.com/david-cortes/fneighcf',
  download_url = 'https://github.com/david-cortes/fneighcf/archive/0.2.2.tar.gz',
  keywords = ['collaborative filtering', 'item-item similarity', 'factored neighborhood'],
  classifiers = [],

  cmdclass = {'build_ext': build_ext},
  ext_modules = [ext_mod,
    ]
)
