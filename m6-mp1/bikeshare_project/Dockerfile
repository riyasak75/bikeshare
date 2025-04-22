# Use an official Python base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory in the container
WORKDIR /app

# Copy project files into the container
COPY . .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install fastapi uvicorn[standard]
RUN pip install -e .  # installs bikeshare_model as a package
RUN pip install -r requirements/requirements.txt
RUN pip install pydantic-settings


# Expose the port your app runs on
EXPOSE 8000

# Run the app using uvicorn
CMD ["uvicorn", "bike_sharing_api.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]


