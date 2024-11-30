"""Copyright: Nabarun Goswami (2023)."""
from setuptools import setup, find_packages

setup(
    name='custom_hf_trainer',
    version='0.1.1',
    author='Nabarun Goswami',
    author_email='nabarungoswami@mi.t.u-tokyo.ac.jp',
    description='A custom Hugging Face trainer for logging auxiliary losses',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/naba89/custom_hf_trainer',
    packages=find_packages(),
    install_requires=[
        'torch',
        'transformers==4.46.3',
        'accelerate==1.1.1',
        'datasets==3.1.0',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)
