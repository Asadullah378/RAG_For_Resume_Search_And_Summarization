# Use an official Python runtime as the parent image
FROM python:3.8-slim

# Set the working directory in the docker image
WORKDIR /app

COPY requirements.txt /app

# Install the required packages
RUN pip install -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app

# Make port 4000 available to the world outside this container
EXPOSE 4000

# Run Application
CMD ["python", "app.py"]