# Define function directory
#ARG FUNCTION_DIR="/function"

# download image
FROM public.ecr.aws/lambda/python:3.6

#FROM arajesh17/automorph_lee:v1

# Install aws-lambda-cpp build dependencies
#RUN apt-get update && \
#  apt-get install -y \
#  g++ \
#  make \
#  cmake \
#  unzip \
#  libcurl4-openssl-dev \
#  libgl1-mesa-dev

# Create function directory
#RUN mkdir -p ${FUNCTION_DIR}

# Copy function code
COPY app/lambda_predict.py ./
COPY requirements.txt ./
COPY setup.py ./

# Can remove the ptvsd install
RUN pip install --upgrade pip && pip install -r requirements.txt

RUN pip install ptvsd

COPY automorph/ ./automorph
RUN pip install automorph/

# Install the runtime interface client
#RUN pip install awslambdaric

#ENTRYPOINT [ "/opt/conda/envs/automorph/bin/python", "-m", "awslambdaric" ]
CMD ["lambda_predict.lambda_handler"]