from setuptools import setup, find_packages
# print(find_packages())
setup(
    name='latentrl',
    version='0.1.0',
    author='Xue Yuan',
    description="package to explore rl and vqvae",
    # packages=find_packages(include=['models', 'utils']),
    packages=[package for package in find_packages() if package.startswith("latentrl")],
    # packages=find_packages(),
    # package_dir = {'': 'latentrl'},
    install_requires=[
        'PyYAML',
        'numpy'
    ],
    extras_require={
        'interactive': ['jupyter',]
    }
)