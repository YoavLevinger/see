```python
import psycopg2

# Create connection to the PostgreSQL database
conn = psycopg2.connect(
    dbname="mydb",
    user="username",
    password="password",
    host="localhost"
)

cur = conn.cursor()

# Create table for users
cur.execute("""
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL
);
""")

# Insert user into users table
cur.execute("INSERT INTO users (username, email, password) VALUES (%s, %s, %s)", ("john_doe", "john@example.com", "password123"))

# Update user's password in the users table
cur.execute("UPDATE users SET password=%s WHERE username=%s", ("new_password", "john_doe"))

# Query data from the users table
cur.execute("SELECT * FROM users")
rows = cur.fetchall()
for row in rows:
    print(row)

# Commit and close connection
conn.commit()
cur.close()
conn.close()
```