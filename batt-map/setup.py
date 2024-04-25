from setuptools import setup, find_packages

# with open("README.md", "r",encoding="utf8") as fh:
#     long_description = fh.read()

setup(name='batt-map',
      version='0.1',
      description='Utilities for battery mapping analysis',
      # long_description=long_description,
      # long_description_content_type="text/markdown",
      author='Jake Huang',
      author_email='jdhuang@mines.edu',
      license='BSD 3-clause',
      packages=find_packages(),
      install_requires=[
          'numpy',
          'pandas',
          'scipy',
          'matplotlib',
      ],
      include_package_data=True
      )
