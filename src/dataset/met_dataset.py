import os
import requests
import time

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../../'))
import src.config


# URL for the Met Collection API
base_url = "https://collectionapi.metmuseum.org/public/collection/v1"

# Function to get objects from the Met's Costume Institute by decade
def get_images_met(decade_start, decade_end):
    search_url = f"{base_url}/search"
    queries = ["dress", "corset", "bodice", "vest", "suit", "waistcoat", "trousers", "coat", "skirt", "cape", "underwear"]

    unique_object_ids = set()
    object_ids = []
    image_data = []
    duplicates = 0

    for query in queries:
        # Search for objects within the costume institute department and decade range
        params = {
            "department": "Costume Institute",
            "dateBegin": decade_start,
            "dateEnd": decade_end,
            "hasImage": True,
            "medium": "Cotton" or "Silk" or "Wool" or "Linen",
            "q": query,
        }

        response = requests.get(search_url, params=params)
        data = response.json()
        print(f"Acquired data for query {query}.")

        if not data.get('objectIDs'):
            print(f"No objects found for {decade_start} to {decade_end} searching for {query}")
        else:
            print(f"Found {len(data['objectIDs'])} objects for {decade_start} to {decade_end} searching for {query}")

            # Filter duplicates and count removed entries
            for obj_id in data['objectIDs']:
                if obj_id not in unique_object_ids:
                    unique_object_ids.add(obj_id)  
                    object_ids.append(obj_id)  
                else:
                    duplicates += 1

    print(f"Total unique objects found: {len(object_ids)}")
    print(f"Duplicate objects removed: {duplicates}")

  
    request_count = 0
    obj_with_img = 0

    # Fetch image url for each object id
    # Limit to 80 requests per sec - api max handling
    for obj_id in object_ids:
        object_url = f"{base_url}/objects/{obj_id}"
        obj_response = requests.get(object_url)
        obj_data = obj_response.json()

        # Only entries with a primary image are usable
        if obj_data.get("primaryImage"):
            image_info = {
                "objectID": obj_id,
                "primaryImage": obj_data["primaryImage"],
            }

            image_data.append(image_info)
            obj_with_img += 1


        request_count += 1

        if request_count % 80 == 0:
            time.sleep(1)  
    print(f"Total complete entries to save: {obj_with_img}")
    return image_data




# Save images to a local folder
def save_images(image_data, folder_name, decade):
    folder_name = f"{folder_name}/{decade}s"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    for idx, item in enumerate(image_data):

        image_url = item['primaryImage']

        image_name = f"{folder_name}/{idx}_{item['objectID']}.jpg"

        # Download the image
        img_data = requests.get(image_url).content
        with open(image_name, 'wb') as handler:
            handler.write(img_data)

        print(f"Saved: {image_name}")


if __name__ == "__main__":
    all_images = []

    for decade in range(1830, 1970, 10):
        decade_start = decade
        decade_end = decade + 9
        image_data = get_images_met(decade_start, decade_end)
        save_images(image_data, src.config.MET_DATA_DIR, decade)
        all_images.extend(image_data)

    print(f"Saved {len(all_images)} images to {src.config.MET_DATA_DIR}.")  

