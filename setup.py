from setuptools import setup, find_packages

long_description = """Experimental package to explore financila analysis"""

setup(name='qf',
      version='0.1',
      description='Quantative Finance library',
      author='Yevgeniy Yermoshin',
      author_email='yev.developer@gmail.com',
      license='Apache 2.0',
      long_description=long_description,
      keywords=['pandas', 'data', 'analysis', 'fixed income', 'stocks', 'bond', 'equities', 'timeseries', 'quantative', 'strategies', 'backtesting'],
      url='',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False)
