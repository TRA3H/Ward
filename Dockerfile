# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy model file from the local directory
COPY ./checkpoints/final_model.pth /app/final_model.pth

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV PORT 5000

# Run api.py when the container launches
CMD ["gunicorn", "-b", "0.0.0.0:5000", "api:app"]

