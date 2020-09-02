# Use the official image as a parent image.
FROM python:3.6-slim

# Set the working directory.
WORKDIR /opt/ml/code/

# Copy the file from your host to your current location.
COPY . .

# Run the command inside your image filesystem.
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    gcc swig python3-dev libgl1-mesa-glx libglib2.0-0 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install -r ./requirements.txt
RUN pip3 install sagemaker-training

# Run the specified command within the container.
CMD TIMESTEPS=1e2 python ./main.py
# ENV TIMESTEPS 1e5
# ENV SAGEMAKER_PROGRAM ./train_climber.py

