from setuptools import setup, find_packages

setup(
    name='pyoptimize',
    version='0.0.3',
    author='Andrew Barisser',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        "numpy>=1.15.0",
        ],
    tests_require=['pytest-cov', 'pytest'])
