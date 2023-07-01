# Dockerfile

# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container to /app
WORKDIR /app

# copy requirements first
COPY requirements.txt /app/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Add current directory files to /app in the container
ADD . /app

# Expose port 5000 for the Flask application to run on
EXPOSE 5000

# Run the command when the container launches
CMD ["python", "app.py"]
