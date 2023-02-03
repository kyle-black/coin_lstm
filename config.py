import boto3


s3 = boto3.resource(
    service_name='s3',
    region_name='us-east-2',
    aws_access_key_id= 'AKIAWBYWRIYXTFZE5QRY',
    aws_secret_access_key='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'
)






for bucket in s3.buckets.all():
    print(bucket.name)