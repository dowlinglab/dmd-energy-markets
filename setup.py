from setuptools import setup
import setuptools

with open('./requirements.txt', 'r') as f:
    requirements = f.readlines()

setup(
    name='dmdTrading',
    version='0.1',
    description='Code used for DMD forecasting with energy market datasets',
    author='Clay Elmore',
    author_email='celmore25@gmail.com',
    license='GNU LESSER GENERAL PUBLIC LICENSE',
    packages=setuptools.find_packages(),
    include_package_data=True,
    scripts=[],
    zip_safe=False,
    install_requires=requirements,
)
