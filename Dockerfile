FROM python:3.9.7-slim

# Install pipenv and required system packages
RUN apt-get update && apt-get install -y \
    libgl1-mesa-dev \
    libglib2.0-0 \
    && apt-get clean \
    && pip install pipenv

# Workdir
WORKDIR /app

# Copy Pipfile and Pipfile.lock files in working dir
COPY ["Pipfile", "Pipfile.lock", "./"]

# Install required packages using pipenv
RUN pipenv install --system --deploy

# Copy utils folder in working dir
COPY ["utils", "./utils"]

# Copy templates folder, static folder in working dir
COPY ["templates", "./templates"]

# Copy static folder in working dir
COPY ["static", "./static"]

# Copy models folder in working dir
COPY ["models", "./models"]

# copy main.py in the working dir
COPY ["main.py", "./"]

# Expose port 5000
EXPOSE 5000

# Bind the app to
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:5000", "main:app"]