"""
Check collection status and attributes
"""

import requests
from config import Config

def check_collection():
    """Check the buses collection status"""
    
    headers = {
        'X-Appwrite-Project': Config.APPWRITE_PROJECT_ID,
        'X-Appwrite-Key': Config.APPWRITE_API_KEY,
        'Content-Type': 'application/json'
    }
    
    # Get collection info
    collection_url = f"{Config.APPWRITE_PUBLIC_ENDPOINT}/databases/{Config.DATABASE_ID}/collections/{Config.BUS_COLLECTION_ID}"
    
    try:
        response = requests.get(collection_url, headers=headers)
        print(f"Collection Status: {response.status_code}")
        if response.status_code == 200:
            print(f"Collection Info: {response.json()}")
    except Exception as e:
        print(f"Error getting collection: {e}")
    
    # Get attributes
    attributes_url = f"{Config.APPWRITE_PUBLIC_ENDPOINT}/databases/{Config.DATABASE_ID}/collections/{Config.BUS_COLLECTION_ID}/attributes"
    
    try:
        response = requests.get(attributes_url, headers=headers)
        print(f"\nAttributes Status: {response.status_code}")
        if response.status_code == 200:
            attributes = response.json()
            print("Current Attributes:")
            for attr in attributes.get('attributes', []):
                print(f"  - {attr['key']}: {attr['type']} (status: {attr['status']})")
    except Exception as e:
        print(f"Error getting attributes: {e}")

if __name__ == "__main__":
    check_collection()
