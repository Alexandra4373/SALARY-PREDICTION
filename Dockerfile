# Start from a small official Python image
FROM python:3.14-slim
# All following commands run inside /code
WORKDIR /code
# Copy the package list first so Docker can cache the install step
COPY requirements-docker.txt .
# Install the packages
RUN pip install --no-cache-dir -r requirements-docker.txt
# Copy the app code and the trained model
COPY app/ app/
COPY model/ model/
# Start the server on port 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
