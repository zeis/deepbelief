from setuptools import setup, find_packages

setup(
    name='deepbelief',
    version='1.0.0.dev0',
    packages=find_packages(exclude=['scripts', 'tests']),
    test_suite="tests",
    install_requires=['numpy', 'scipy', 'matplotlib', 'h5py', 'pillow']
)
