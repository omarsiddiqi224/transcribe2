# Python image to use.
FROM python:3.9

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file used for dependencies
COPY requirements.txt .

RUN pip install cython

# Install ffmpeg
RUN apt-get update && apt-get install -y ffmpeg

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

RUN pip install ctransformers

RUN pip install accelerate



EXPOSE 8080

# Copy the rest of the working directory contents into the container at /app
COPY . .

# Run app.py when the container launches
ENTRYPOINT ["python", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
