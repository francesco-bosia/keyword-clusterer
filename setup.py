#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [ ]

test_requirements = [ ]

setup(
    author="Francesco Bosia",
    author_email='francesco.bosia@hudsongoodman.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="This projects clusters a set of keywords given as input in an excel table and returns an excel table with the same columns as the input and the index of the clustering.",
    entry_points={
        'console_scripts': [
            'keyword_clustering_with_transformers=keyword_clustering_with_transformers.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='keyword_clustering_with_transformers',
    name='keyword_clustering_with_transformers',
    packages=find_packages(include=['keyword_clustering_with_transformers', 'keyword_clustering_with_transformers.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/francesco-bosia/keyword_clustering_with_transformers',
    version='0.1.0',
    zip_safe=False,
)
