import geopandas as gpd
import requests
import os

# Base URL of the GeoPackage files
base_url = "http://geoglows-v2.s3-website-us-west-2.amazonaws.com/streams/"

# List of GeoPackage file names (you can add more file names to this list)
gpkg_files = [
    "streams_101.gpkg",
    "streams_102.gpkg",
    "streams_103.gpkg",
    "streams_104.gpkg",
    "streams_105.gpkg",
    "streams_106.gpkg",
    "streams_107.gpkg",
    "streams_108.gpkg",
    "streams_109.gpkg",
    # Add more file names as needed
]


# Function to download the GeoPackage file and save it locally
def download_gpkg(url, save_path):
    response = requests.get(url)
    response.raise_for_status()  # Check if the request was successful
    with open(save_path, 'wb') as file:
        file.write(response.content)
    print(f"Downloaded and saved file from {url} to {save_path}")


# Function to search for a specific LINKNO value in a GeoPackage file
def search_linkno_in_gpkg(file_path, linkno_value):
    # Load the GeoPackage file into a GeoDataFrame
    gdf = gpd.read_file(file_path)
    # print(gdf.head())  # Print the first few rows of the GeoDataFrame; commented since I don't think it's needed

    # Search for the LINKNO value in the GeoDataFrame
    result = gdf[gdf['LINKNO'] == linkno_value]

    return not result.empty


# Main function
def main():
    linkno_value = 12003308  # Replace with the specific LINKNO value you are looking for

    for file_name in gpkg_files:
        try:
            url = os.path.join(base_url, file_name)
            save_path = os.path.join("./downloads", file_name)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            download_gpkg(url, save_path)

            if search_linkno_in_gpkg(save_path, linkno_value):
                print(f"LINKNO {linkno_value} found in file: {file_name}")
                break
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")
    else:
        print(f"LINKNO {linkno_value} not found in any of the specified files.")


if __name__ == "__main__":
    main()