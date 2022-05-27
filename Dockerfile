#Install Python
FROM yukundocker/image_automorph

#install boto3
RUN pip install boto3

# copy required files
COPY automorph/ ./automorph
COPY setup.py ./
COPY lambda_predict ./

RUN conda activate automorph
RUN apt-get update && apt-get install -y libgl1-mesa-dev

# install automorph - make sure that I am installing correctly 
RUN pip install -e .

# Set entry point
CMD ["lambda_predict.lambda_handler"]