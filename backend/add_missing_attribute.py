"""
Add missing attribute to existing collection
"""

import requests
from config import Config

def add_missing_attribute():
    """Add the missing current_location attribute to buses collection"""
    
    headers = {
        'X-Appwrite-Project': Config.APPWRITE_PROJECT_ID,
        'X-Appwrite-Key': Config.APPWRITE_API_KEY,
        'Content-Type': 'application/json'
    }
    
    url = f"{Config.APPWRITE_PUBLIC_ENDPOINT}/databases/{Config.DATABASE_ID}/collections/{Config.BUS_COLLECTION_ID}/attributes/string"
    
    data = {
        "key": "current_location",
        "required": False,
        "size": 255
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code in [200, 201]:
            print("✅ current_location attribute added successfully!")
        else:
            print(f"⚠️ Attribute may already exist: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    add_missing_attribute()
