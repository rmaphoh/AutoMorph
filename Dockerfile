#Install Python
FROM public.ecr.aws/lambda/python:3.6

#install poetry
RUN pip install -r requirements.txt 

# copy required files
COPY automorph/ ./

# install automorph 
RUN python setup.py install

# Set entry point
CMD ["lambda_predict.lambda_handler"]