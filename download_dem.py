import requests
import os

def sanitize_filename(filename):
    """Replace or remove invalid characters from a file name."""
    invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    for char in invalid_chars:
        filename = filename.replace(char, "_")  # Replace invalid characters with '_'
    return filename

def search_and_download_dems(lat_long, output_folder):
    lat = lat_long[0]
    long = lat_long[1]
    # bounding box eventually 5 channel widths downstream
    bbox = (lat - 0.005, long - 0.005, lat + 0.005, long + 0.005)

    products = [
        "Digital Elevation Model (DEM) 1 meter",
        "National Elevation Dataset (NED) 1/9 arc-second",
        "National Elevation Dataset (NED) 1/3 arc-second Current"
    ]

    base_url = "https://tnmaccess.nationalmap.gov/api/v1/products"

    for product in products:
        # Define the API query parameters
        params = {
            "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
            "datasets": product,
            "max": 10,  # Number of results to return
            "outputFormat": "JSON"
        }

        try:
            # query the api
            response = requests.get(base_url, params=params)
            response.raise_for_status()

            # parse the results
            results = response.json().get("items", [])
            if not results:
                print(f"No results found for {product} data.")
                continue

            print(f"Found {len(results)} results for {product} data.")

            os.makedirs(output_folder, exist_ok=True)

            for item in results:
                title = item.get("title", "Unnamed")
                sanitized_title = sanitize_filename(title)  # Sanitize the file name
                download_url = item.get("downloadURL")

                if download_url:
                    local_filename = os.path.join(output_folder, f"{sanitized_title}.tiff")
                    print(f"Downloading {sanitized_title}...")

                    with requests.get(download_url, stream=True) as r:
                        r.raise_for_status()
                        with open(local_filename, "wb") as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                f.write(chunk)
                    print(f"Saved to {local_filename}")
                else:
                    print(f"No download URL for {title}. Skipping...")

            # If we successfully downloaded data for the current product, we can stop
            break

        except requests.RequestException as e:
            print(f"Error occurred: {e}")


# 1/9 arc-second example
"""
maple = [-97.147827, 46.798457]
output_loc = "./maple"
search_and_download_dems(maple, output_loc)
"""