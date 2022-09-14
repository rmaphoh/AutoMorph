# Code for AWS INSTALL
## download image
#FROM public.ecr.aws/lambda/python:3.6
#
## Copy function code
#COPY app/lambda_predict.py ./
#COPY requirements.txt ./
#COPY setup.py ./
#
#RUN pip install --upgrade pip && pip install -r requirements.txt
#
#COPY automorph/ ./automorph
#RUN pip install automorph/
#
#ENV TORCH_HOME=/tmp
#
#CMD ["lambda_predict.lambda_handler"]

#AZURE attempt
#FROM 10.0.0.95:5000/arajesh/automorph_lee
FROM python:3.6

# Copy function code
COPY requirements.txt ./
COPY setup.py ./
#
RUN pip install --upgrade pip && pip install -r requirements.txt
#
##cv2 dependencies
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

COPY automorph/ ./automorph
RUN python setup.py install

