FROM homeassistant/amd64-base:latest

# Install necessary packages
RUN apk update && apk add --no-cache \
    python3 \
    py3-pip \
    bash \
    && pip3 install flask flask-cors

# Copy the app files including requirements.txt
COPY plant_care /app/plant_care
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip3 install -r /app/requirements.txt

# Expose the port the app will run on
EXPOSE 80

# Set the working directory
WORKDIR /app/plant_care

# Start the Flask app
CMD ["python3", "plant_care_local.py"]
