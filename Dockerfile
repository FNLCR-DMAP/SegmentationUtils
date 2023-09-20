FROM continuumio/miniconda3

# Copy the requirements file
COPY requirements.txt ./requirements.txt

# Create a new environment
RUN conda create -n env python=3.11

# Activate the environment
RUN echo "source activate env" > ~/.bashrc
ENV PATH /opt/conda/envs/env/bin:$PATH

# Install the requirements
RUN pip install -r requirements.txt