import requests
from met_dataset import save_images
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../../'))
import config.config



def get_images_va(decade_start, decade_end):
    va_base_url = "https://api.vam.ac.uk/v2/objects/search"
    image_data = []
    unique_object_ids = set()
    duplicates = 0


    types = ["Dress", "Waistcoat", "Blouse", "Jacket", "Shawl"]

    for type in types:
        params = {
            "id_category": "THES48975",
            "id_collection": "THES48601",
            "year_made_from": decade_start,
            "year_made_to": decade_end,
            "has_images": True,
            "kw_object_type": type
        }

        response = requests.get(va_base_url, params=params)
        data = response.json()

        if 'records' in data:
            print(f"Found {len(data['records'])} objects for {decade_start} to {decade_end} searching for {type}")

            for record in data['records']:

                # Filter duplicates
                if record['systemNumber'] not in unique_object_ids and record['_primaryImageId']:
                    unique_object_ids.add(record['systemNumber'])
                    image_info = {
                        "objectID": record['systemNumber'],
                        "primaryImage": f"https://framemark.vam.ac.uk/collections/{record['_primaryImageId']}/full/full/0/default.jpg",

                    }
                    image_data.append(image_info)

                else:
                    duplicates += 1
        else:
            print(f"No objects found for {decade_start} to {decade_end} searching for {type}")


    print(f"Total unique objects found: {len(image_data)}")
    print(f"Duplicate objects removed: {duplicates}")

    return image_data


if __name__ == "__main__":
    all_images = []

    for decade in range(1830, 1970, 10):
        decade_start = decade
        decade_end = decade + 9
        image_data = get_images_va(decade_start, decade_end)
        save_images(image_data, config.config.VA_DATA_DIR, decade)
        all_images.extend(image_data)

    print(f"Saved {len(all_images)} images to {config.config.VA_DATA_DIR}.")  