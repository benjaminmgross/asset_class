import os
from setuptools import setup, find_packages


def read(fname):
    try:
        with open(os.path.join(os.path.dirname(__file__), fname)) as fh:
            return fh.read()
    except IOError:
        return ''

requirements = read('REQUIREMENTS').splitlines()
tests_requirements = read('REQUIREMENTS-TESTS').splitlines()

setup(
    name="asset_class",
    version="0.0.1",
    description="",
    long_description=read('README.rst'),
    url='https://github.com/benjaminmgross/asset_class',
    license='',
    author='Benjamin M. Gross',
    author_email='benjaminMgross@gmail.com',
    packages=['asset_class'],
    include_package_data=True,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python',
    ],
    install_requires=requirements,
    tests_require=tests_requirements,
)
