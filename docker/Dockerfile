FROM continuumio/anaconda3

LABEL name="fda" \
summary="financial-data-analysis Python Docker image" \
description="financial-data-analysis Python Docker image" \
maintainer="yevgeniy.yermoshin@db.com"

ENV USER_NAME=fda-user

ENV USER_ID=1000

ARG YUM_DEPS="bzip2 gcc gcc-c++ nmap-ncat git krb5-devel"

ARG CONDA_VERSION=""


ARG PIP-INDEX_URL=""

ENV APP-NAME="fda"
ENV HOME="/home/${USERNAME}"
ENV CONDA_HOME="/opt/Conda-${CONDA—VERSION}"
ENV PATH="${CONDA_HOME}/bin:${PATH}"
ENV VENV_HOME="/opt/python/venv"
ENV PATH="$VENV—HOME/bin:$PATH"
ENV PYTHONUNBUFFERED=1

RUN useradd -u ${USERID} --shell /bin/bash --gid 0 --create-home --home-dir ${HOME} ${USER_NAME}build

RUN echo "Installing Yum dependencies" && \
yum install ${YUM_DEPS} -y --skip-broken && \
yum clean all && \
rm -rf /var/cache/yum

WORKDIR ${HOME}

COPY fda ${HOME}/fda

COPY requirements.txt ${HOME}/requirements.txt

COPY setup.py ${HOME}/setup.py

RUN echo "Creating & activating virtual environment" && \
python -m venv ${VENV HOME} && \
source ${VENV_HOME}/bin/activate && \
echo "Configuring Pip" && \
pip install --index-url ${PIPINDEXURL} --upgrade pip==21.0.1 && \
echo "Installing Pip dependencies" && \
pip install --index-url ${PIPINDEXURL} -r requirements.txt && \
pip install -e .


RUN echo "Creating Jupyter notebook related directories" && \
mkdir -p ${HOME}/notebooks && \
mkdir -p ${HOME}/notebooks/data

RUN echo "Setting up Jupyter password" && \
jupyter notebook --generate-config && \
echo "c.NotebookApp.password='shal;***'"»${home}/.jupyter/jupyternotebookconfig.py

CMD [ "jupyter","lab", "--notebook-dir=notebooks", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.token=''","--NotebookApp.password=''"]







