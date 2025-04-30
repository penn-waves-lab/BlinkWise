## Use the official TensorFlow GPU image as the base image
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Set environment variables to avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/New_York

# Set the working directory in the container
WORKDIR /app

# Copy only the requirements file
COPY requirements.txt /app/requirements.txt

# Install Graphviz and Python 3
RUN apt-get update && apt-get install -y graphviz python3 python3-pip

# create an alias for python3
RUN ln -sf /usr/bin/python3 /usr/bin/python

# install tensorflow
RUN pip install --no-cache-dir tensorflow[and-cuda]==2.15.0.post1

# install jupyter
RUN pip install --no-cache-dir jupyterlab

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Set the entrypoint to the run script
ENTRYPOINT ["/bin/bash"]

# Default CMD is empty, allowing for easy overriding
CMD []