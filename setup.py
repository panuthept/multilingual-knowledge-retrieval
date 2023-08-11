from setuptools import setup, find_packages


with open("requirements.txt") as f:
    dependencies = [line for line in f][:-1]

setup(
    name='mkr',
    version='1.0',
    packages=find_packages("src"),
    package_dir={'': 'src'},
    license='MIT License',
    author='Panuthep Tasawong',
    author_email='panuthep.t_s20@vistec.ac.th',
    description='Multilingual Knowledge Retrieval',
    python_requires='>=3.7',
    install_requires=dependencies
)