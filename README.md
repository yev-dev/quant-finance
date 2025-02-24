
# Experimental project expore statistical and ML techniques for financial analysis. 

### Installation

Python dependencies are managed by [pip-tools](https://pypi.org/project/pip-tools/). You need to create conda or venv for your env first.

pip-compile generates a requirements.txt file using the latest versions that fulfil the dependencies you specify in the supported files.

If pip-compile finds an existing requirements.txt file that fulfils the dependencies then no changes will be made, even if updates are available.

To force pip-compile to update all packages in an existing requirements.txt, run pip-compile --upgrade.


1. Prepapre Python env 

   Create conda environment from env file:

        conda env create -f environment.yml

    Alternatively, create create a virtual envrionment with 'venv'

        python3 -m venv env
        

2. Activate the environment:

        conda activate qf

        source env/bin/activate

pip-compile requirements.in

4. Install time-series-analysis in editable mode for active development

        pip install -e .

5. Install as a package

        pip install

6. If a new package is added to the requirements.txt file:
   

        pip install --upgrade -r requirements.txt

7. Removing installed virtual environment

    For conda:

        conda remove --name qf --all

    For pip env - delete associated directory

8. Updating with new dependenciespip-compile --upgrade

        pip-compile --upgrade
        
        pip install --ignore-installed -r requirements.txt

        Or

        pip install --upgrade --force-reinstall -r requirements.txt