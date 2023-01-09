from distutils.core import setup
from setuptools import find_packages

setup(
    name='1R2R',
    packages=find_packages(),
    version='0.0.1',
    description='One Risk to Rule Them All: A Risk-Sensitive Perspective on Model-Based Offline Reinforcement Learning',
    long_description=open('./README.md').read(),
    author='',
    author_email='',
    entry_points={
        'console_scripts': (
            '1R2R=_1R2R.scripts.console_scripts:main'
        )
    },
    requires=(),
    zip_safe=True,
    license='MIT'
)
