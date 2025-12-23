# 1. Base Image: Start with a lightweight version of Python 3.9
FROM python:3.9-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy just the requirements first (Optimization technique)
COPY requirements.txt .

# 4. Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your code into the container
COPY . .

# 6. Default command: Drop into a bash shell
CMD ["bash"]
