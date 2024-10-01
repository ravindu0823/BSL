# Use an official Python runtime as a parent image
FROM python:3.11.9

# Set the working directory inside the container
WORKDIR /python-docker

# Install system dependencies needed for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt and install any needed packages specified in it
COPY requirements.txt requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose port 5000 for Flask
EXPOSE 5000:5000

# Define the command to run the Flask application
CMD [ "python", "-m" , "app-production"]
