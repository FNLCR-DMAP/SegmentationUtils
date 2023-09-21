FROM continuumio/miniconda3

# Install utilities
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Create a new environment
RUN conda create -n env python=3.11
RUN conda install -c anaconda git

# Activate the environment
RUN echo "source activate env" > ~/.bashrc
ENV PATH /opt/conda/envs/env/bin:$PATH

# Copy the requirements file
COPY requirements.txt ./requirements.txt

# Install the requirements
RUN pip install -r requirements.txt

CMD ["bash"]