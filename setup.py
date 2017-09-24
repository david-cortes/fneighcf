from distutils.core import setup
setup(
  name = 'fneighcf',
  packages = ['fneighcf'],
  install_requires=[
   'pandas>=0.18.0',
   'numpy'
],
  version = '0.1',
  description = 'Recommender system based on parameterized Item-Item effects',
  author = 'David Cortes',
  author_email = 'david.cortes.rivera@gmail.com',
  url = 'https://github.com/david-cortes/fneighcf',
  download_url = 'https://github.com/david-cortes/fneighcf/archive/0.1.tar.gz',
  keywords = ['collaborative filtering', 'item-item similarity', 'factored neighborhood'],
  classifiers = [],
)