import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

user = os.getenv("NEO4J_USER")
password = os.getenv("NEO4J_PASSWORD")
uri = os.getenv("NEO4J_URI")

assert user and password and uri, "Missing Neo4j environment variables!"

with GraphDatabase.driver(uri, auth=(user, password)) as driver:
    try:
        driver.verify_connectivity()
        print("WSL is successfully talking to Windows Neo4j!")
    except Exception as e:
        print(f"Connection failed: {e}")