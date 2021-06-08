from setuptools import setup, find_packages

setup(
    name='deepbelief',
    version='1.0.0.dev0',
    packages=find_packages(exclude=['examples', 'tests']),
    test_suite="tests",
    install_requires=[
        'h5py==2.9.0',
        'Keras==2.2.4',
        'matplotlib==3.0.3',
        'numpy==1.16.4',
        'Pillow==8.2.0',
        'scipy==1.1.0',
        'six==1.12.0'
    ]
)
