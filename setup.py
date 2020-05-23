from setuptools import setup, find_packages


VERSION = '0.0.7'

with open('requirements.txt') as f:
    install_requires = f.read().strip().split('\n')


setup(
    name='eisen-core',
    version=VERSION,
    description='Eisen is a collection of tools to train neural networks for medical image analysis',
    packages=find_packages(),
    install_requires=install_requires,
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [],
    },
)