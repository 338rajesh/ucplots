from setuptools import setup, find_packages

setup(
    name='ucplots',
    version='0.2.0',
    author='Rajesh Nakka',
    author_email='338rajesh@gmail.com',
    description='A simple package for plotting unit cells',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)
