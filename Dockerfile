# Lightweight python version
FROM python:3.10-slim

# Set the working directory to the project root
WORKDIR /app

# Copy the entire project into the container
COPY . .

# Install dependencies using the requirements file inside the dashboard folder
RUN pip install --no-cache-dir -r dashboard/requirements.txt

# Hugging Face Spaces requires port 7860
EXPOSE 7860

# Change directory into the dashboard folder so relative paths work
WORKDIR /app/dashboard

# Command to run the Streamlit application
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]