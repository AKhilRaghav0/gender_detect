"""
Fix the current_location attribute to handle JSON objects
"""

import requests
from config import Config

def fix_location_attribute():
    """Fix the current_location attribute to handle JSON objects"""
    
    headers = {
        'X-Appwrite-Project': Config.APPWRITE_PROJECT_ID,
        'X-Appwrite-Key': Config.APPWRITE_API_KEY,
        'Content-Type': 'application/json'
    }
    
    # First, delete the existing string attribute
    delete_url = f"{Config.APPWRITE_PUBLIC_ENDPOINT}/databases/{Config.DATABASE_ID}/collections/{Config.BUS_COLLECTION_ID}/attributes/current_location"
    
    try:
        response = requests.delete(delete_url, headers=headers)
        print(f"Delete current_location attribute: {response.status_code}")
        if response.status_code == 204:
            print("✅ current_location attribute deleted")
        else:
            print(f"⚠️ Delete response: {response.text}")
    except Exception as e:
        print(f"❌ Error deleting attribute: {e}")
    
    # Wait a moment for deletion to complete
    import time
    time.sleep(2)
    
    # Now recreate it as a proper attribute for JSON objects
    # For Appwrite, we'll use a larger string field that can hold JSON
    create_url = f"{Config.APPWRITE_PUBLIC_ENDPOINT}/databases/{Config.DATABASE_ID}/collections/{Config.BUS_COLLECTION_ID}/attributes/string"
    
    data = {
        "key": "current_location",
        "required": False,
        "size": 1000  # Larger size to accommodate JSON
    }
    
    try:
        response = requests.post(create_url, headers=headers, json=data)
        print(f"Create current_location attribute: {response.status_code}")
        if response.status_code in [200, 201]:
            print("✅ current_location attribute recreated successfully!")
        else:
            print(f"⚠️ Create response: {response.text}")
    except Exception as e:
        print(f"❌ Error creating attribute: {e}")

if __name__ == "__main__":
    fix_location_attribute()
