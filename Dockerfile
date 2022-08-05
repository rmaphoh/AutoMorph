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

#download image
FROM python:3.6

# Copy function code
COPY app/lambda_predict.py ./
COPY requirements.txt ./
COPY setup.py ./

RUN pip install --upgrade pip && pip install -r requirements.txt

COPY automorph/ ./automorph
RUN pip install automorph/

#torch home this had to be changed for AWS
#ENV TORCH_HOME=''

#CMD ['main.main']