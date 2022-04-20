"""
Setup of Project 3 python codebase
Authors: Chris Corea, Jay Monga
"""
from setuptools import setup

requirements = [
    'trimesh[easy]',
    'vedo',
    'numpy',
    'scipy',
    'shapely',
    'pyyaml',
    'casadi',
    'rtree'
]

setup(name='proj3_pkg',
      version='0.2.0',
      description='C106B Grasping Lab project code',
      author='Jay Monga',
      author_email='jay.monga16@berkeley.edu',
      package_dir = {'': 'src'},
      packages=['joint_trajectory_action_server', 'utils', 'metrics', 'policies'],
      install_requires=requirements,
      test_suite='test'
     )
