# download image
FROM public.ecr.aws/lambda/python:3.6

# Copy function code
COPY app/lambda_predict.py ./
COPY requirements.txt ./
COPY setup.py ./

RUN pip install --upgrade pip && pip install -r requirements.txt

# Can remove the ptvsd install
#RUN pip install ptvsd

COPY automorph/ ./automorph
RUN pip install automorph/

ENV TORCH_HOME=/tmp

#ENTRYPOINT [ "/opt/conda/envs/automorph/bin/python", "-m", "awslambdaric" ]
CMD ["lambda_predict.lambda_handler"]