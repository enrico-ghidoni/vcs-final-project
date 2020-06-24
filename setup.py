from setuptools import setup, find_packages

setup(
    name='g11-vcs-final-project',
    version='0.0.1',
    packages=find_packages(),
    author='Simone Ferrari, Enrico Ghidoni, Tommaso Miana',
    author_email='279007@studenti.unimore.it',
    description='UNIMORE Vision and Cognitive Systems A.A. 2019-2020 Final Project',
    entry_points={
        'console_scripts': [
            'pipeline=progetto.pipeline:pipeline_entry_point'
        ]
    }
)