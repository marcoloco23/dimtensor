# Container image for AWS Lambda with dimtensor
# Use this for better cold start performance vs zip deployment

FROM public.ecr.aws/lambda/python:3.12

# Copy requirements
COPY requirements.txt .

# Install dependencies
# Pin numpy <2.0 for compatibility
RUN pip install --no-cache-dir -r requirements.txt

# Copy function code
COPY app.py ${LAMBDA_TASK_ROOT}/

# Set the CMD to your handler
CMD ["app.lambda_handler"]
