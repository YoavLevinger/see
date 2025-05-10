```python
from datetime import datetime
from peewee import *

db = SqliteDatabase('cloud_usage.db')

class BaseModel(Model):
    class Meta:
        database = db

class CloudProvider(BaseModel):
    provider_id = IntegerField()
    name = CharField()

class Service(BaseModel):
    service_id = IntegerField()
    name = CharField()
    cloud_provider = ForeignKeyField(CloudProvider, backref='services')

class Usage(BaseModel):
    usage_id = IntegerField()
    timestamp = DateTimeField(default=datetime.now)
    service = ForeignKeyField(Service, backref='usages')
    resource = CharField()
    amount = FloatField()
```

This code uses the Peewee ORM for SQLite to define a CloudUsage data model. The `CloudProvider` class represents cloud providers with an id and name. The `Service` class represents services offered by a cloud provider, linking back to the `CloudProvider`. The `Usage` class represents usage of a service, recording the timestamp, resource used, and amount. Each Usage is linked to its corresponding Service.