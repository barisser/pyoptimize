from setuptools import setup, find_packages
#from Cython.Build import cythonize

setup(
    name='pyoptimize',
    version='0.0.2',
    author='Andrew Barisser',
    license='MIT',
    packages=find_packages(),
 #   ext_modules = cythonize("pyoptimize/main.pyx"),
    install_requires=[
  #  	"cython",
        "numpy>=1.15.0",
        ],
    tests_require=['pytest-cov', 'pytest'])
