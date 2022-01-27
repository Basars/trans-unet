from setuptools import find_packages
from setuptools import setup

setup(name='trans-unet',
      version='0.0.2',
      description='An ML model with U-shaped architecture with ResNet50V2 and Vision Transformer based encoders',
      url='https://github.com/Basars/trans-unet.git',
      author='OrigamiDream',
      packages=find_packages(),
      zip_safe=False,
      include_package_data=True,
      install_requires=[
          'tensorflow>=2.0',
          'numpy',
          'scipy'
      ],
      extra_require={
          'tests': ['pytest']
      })
