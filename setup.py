# Python implementation by Dustin Zacharias (2017), method based on Fabrice Rouah (volopta.com)

from setuptools import setup

setup(name='hngoption',
      version='1.1',
      description='Heston Nandi GARCH Option Pricing Model (2000)',
      url='https://github.com/SW71X/hngoption2',
      download_url='https://github.com/SW71X/hngoption2/releases',
      author='Dustin L. Zacharias',
      author_email='zacharias@mit.edu',
      license='MIT',
      packages=['hngoption'],
      install_requires=['pandas'],
      zip_safe=False)

