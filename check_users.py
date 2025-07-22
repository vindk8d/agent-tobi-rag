import asyncio
import sys
import pathlib

# Add backend to path
backend_path = pathlib.Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

from database import db_client

async def main():
    client = await db_client()
    result = await client.execute('SELECT id, username, user_type FROM users LIMIT 5')
    print("Users in database:")
    for row in result:
        print(f"  ID: {row[0]}, Username: {row[1]}, Type: {row[2]}")

if __name__ == "__main__":
    asyncio.run(main()) 