```python
import boto3

def get_resource_usage(resource_type):
    session = boto3.Session(profile_name='your_aws_profile')
    client = session.client('cloudwatch', region_name='us-west-2') # change the region if needed

    namespace = 'AWS/EC2'
    metric = f'CPUUtilization' if resource_type == 'EC2 instance' else 'ElasticIngress' if resource_type == 'Application Load Balancer' else None

    if not metric:
        return None

    start_time = int(datetime.datetime.now(tz=pytz.utc) - datetime.timedelta(days=30)).timestamp()
    end_time = int(datetime.datetime.now(tz=pytz.utc).timestamp())

    response = client.get_metric_statistics(
        Namespace=namespace,
        MetricName=metric,
        Dimensions=[{'Name': 'ResourceId', 'Value': 'your_resource_id'}], # replace with your resource id
        StartTime=start_time,
        EndTime=end_time,
        Period=86400, # 1 day in seconds
        Stat='SampleCount'
    )

    total = response['Datapoints'][0]['SampleCount'] if response['Datapoints'] else None
    return total
```