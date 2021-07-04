import os
import setuptools
from setuptools import setup, find_packages
from distutils.core import setup

# Utility function to read the README file.

def read(fname):
	return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
	name = "eng_mri",
	version = "0.1",
	author = "Jonathan Dunn",
	author_email = "jonathan.dunn@canterbury.ac.nz",
	description = ("Detect code switching (eng/mri)"),
	license = "LGPL 3.0",
	url = "https://github.com/jonathandunn/eng_mri.git",
	keywords = "language id, code switching, mri",
	packages = find_packages(exclude=["*.pyc", "__pycache__"]),
	package_data={'': ['eng_mri.v24.2.NZ.Multi.Ngrams_v1.50k.50x1_layers.*','eng_mri.*']},
	install_requires=["cytoolz",
						"numpy",
						"sklearn",
						"tensorflow",
						],
	include_package_data=True,
	long_description=read('README.md'),
	long_description_content_type='text/markdown',
	)