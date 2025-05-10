```python
import psycopg2

def connect_db():
    try:
        conn = psycopg2.connect(
            dbname="your_database_name",
            user="your_username",
            password="your_password",
            host="your_host",
            port="your_port"
        )
        return conn
    except Exception as e:
        print(f"Error: {e}")
        return None
```