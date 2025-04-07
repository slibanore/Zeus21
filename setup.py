#!/usr/bin/env python

from setuptools import setup, find_packages

# SarahLibanore: install oLIMpus and zeus21
setup(
    name='oLIMpus',
          version='0.1dev',
          description='oLIMpus: cross-correlating lines with Zeus21.',
          url='https://github.com/slibanore/Zeus21',
          author='Sarah Libanore, Julian B. Muñoz, Yonatan Sklansky, Ely Kovetz',
          author_email='libanore@bgu.ac.il',
          #license='MIT',
          packages=['oLIMpus','zeus21'],
          long_description=open('README.md').read(),
          install_requires=[
           "numpy",
           "scipy",
           "mcfit",
           "classy",
           "numexpr",
           "astropy",
           "zeus21"
       ],
)
