# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app except images (dockrignore)
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port 7860 to allow Gradio to be served
EXPOSE 7860

# Command to run your app
CMD ["python", "demo.py"]