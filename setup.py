from setuptools import find_packages, setup
from setuptools import Extension
import numpy as np
from Cython.Build import cythonize

__version__ = "0.0.6"

extensions = [
    Extension(
        name="cjoin", 
        sources=["wombat_db/ops/cjoin.c"], 
        include_dirs=[np.get_include()]
    )
]

with open('README.md') as readme_file:
    README = readme_file.read()

setup(
    name='wombat_db',
    version=__version__,
    description='Useful data crunching tools for pyarrow',
    long_description_content_type="text/markdown",
    long_description=README,
    license='APACHE',
    packages=find_packages(),
    author='Tom Scheffers',
    author_email='tom@youngbulls.nl ',
    keywords=['arrow', 'pyarrow', 'data', 'sql', 'dataframe'],
    url='https://github.com/TomScheffers/wombat',
    download_url='https://pypi.org/project/wombat-db/',

    ext_modules=cythonize(extensions),
    install_requires=[
        'numpy>=1.19.2',
        'pyarrow>=3.0'
    ],
)