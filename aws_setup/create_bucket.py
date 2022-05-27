import boto3

def create_bucket(region:str, bucket_name:str) -> dict:

  s3 = boto3.client('s3')
  response = s3.create_bucket(
                  Bucket=bucket_name,
                          CreateBucketConfiguration={
                                          'LocationConstraint':region
                                                  }
                              )
  return response
  
region = 'us-west-2'
bucket_name = 'automorph-lambda-buckets'
create_bucket(region, bucket_name)
