from setuptools import setup, find_packages
from setuptools import Extension
from distutils.command.build import build as build_orig
import numpy as np

__version__ = "0.0.4"

extensions = [
    Extension(
        name="cjoin", 
        sources=["wombat_db/ops/cjoin.c"], 
        include_dirs=[np.get_include()]
    )
]

with open('README.md') as readme_file:
    README = readme_file.read()

class build(build_orig):
    def finalize_options(self):
        super().finalize_options()
        #__builtins__.__NUMPY_SETUP__ = False
        import numpy
        for extension in self.distribution.ext_modules:
            extension.include_dirs.append(numpy.get_include())
        from Cython.Build import cythonize
        self.distribution.ext_modules = cythonize(self.distribution.ext_modules,
                                                  language_level=3)

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
    include_package_data=True,
    ext_modules=extensions,
    
    # install_requires=["numpy", "pyarrow"],
    # zip_safe=False,
    # setup_requires=["numpy"],
    # package_data={"wombat": ["wombat/ops/cjoin.pyx"]},
    # cmdclass={"build": build},
)