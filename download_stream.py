import os
import geopandas as gpd
import requests
from download_dem import sanitize_filename

def search_and_download_gbd(lat_long, output_folder):
    lat = lat_long[0]
    long = lat_long[1]
    # bounding box eventually 5 channel widths downstream
    bbox = (lat - 0.005, long - 0.005, lat + 0.005, long + 0.005)

    product = "National Hydrography Dataset Plus High Resolution (NHDPlus HR)"

    base_url = "https://tnmaccess.nationalmap.gov/api/v1/products"

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

# National Hydrography Dataset Plus High Resolution (NHDPlus HR)

# Path to the GDB file
gdb_path = "/Users/kennyquintana/Downloads/NHDPLUS_H_1602_HU4_20220412_GDB/NHDPLUS_H_1602_HU4_20220412_GDB.gdb"

# Name of the stream layer
layer_name = "NHDFlowline"

# Read the layer from the GDB file
gdf = gpd.read_file(gdb_path, layer=layer_name)

# Output path for the extracted layer
output_path = "/Users/kennyquintana/Downloads/NHD_test.shp"

# Save the extracted layer to a shapefile
gdf.to_file(output_path)

print(f"Layer {layer_name} has been extracted to {output_path}")