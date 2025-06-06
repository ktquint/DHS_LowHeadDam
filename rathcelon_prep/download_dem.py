import os
import math
import requests
import rasterio
import pandas as pd
from rasterio.merge import merge

def sanitize_filename(filename):
    """
        some files have '/' in their name like "1/9 arc-second," so we'll fix it
    """
    invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    for char in invalid_chars:
        filename = filename.replace(char, "-")  # Replace invalid characters with '_'
    return filename


def meters_to_latlon(lat0, lon0, dx, dy):
    R = 6378137  # radius of Earth in meters (WGS84)
    dlat = dy / R
    dlon = dx / (R * math.cos(math.radians(lat0)))

    new_lat = lat0 + math.degrees(dlat)
    new_lon = lon0 + math.degrees(dlon)
    return float(new_lat), float(new_lon)


def merge_dems(dem_files, output_filename):
    """
        merges a list of DEMs into one and deletes the originals.
        assuming all DEMs are in the same folder.
    """
    dem_dir = os.path.dirname(dem_files[0])
    output_loc = os.path.join(dem_dir, output_filename)

    # open all DEMs and merge those bad boys
    src_files_to_mosaic = [rasterio.open(fp) for fp in dem_files]
    mosaic, out_transform = merge(src_files_to_mosaic)

    # copy metadata from first DEM to the merged one
    out_meta = src_files_to_mosaic[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_transform
    })

    # save new DEM
    with rasterio.open(output_loc, "w", **out_meta) as dest:
        dest.write(mosaic)

    # close all opened files
    for src in src_files_to_mosaic:
        src.close()

    # confirmed new merged file exists and then scrap the old ones
    if os.path.exists(output_loc):
        for fp in dem_files:
            try:
                os.remove(fp)
                print(f"Deleted {fp}")
            except Exception as e:
                print(f"Could not delete {fp}: {e}")
    else:
        print("Merged DEM not found. Original files not deleted.")


def download_dems(lhd_df, dem_dir, resolution):
    """
        this bad boy takes a DataFrame, finds the DEM associated with the lat/lon, records the DEM name in the DataFrame,
        downloads unique DEMs, and returns an updated DataFrame

        lhd_df: DataFrame with lat/lon on each low-head dam
        dem_dir: directory where DEM subdirectories will be located
        resolution: preferred resolution of the DEM
    """
    base_url = "https://tnmaccess.nationalmap.gov/api/v1/products" # all products are found here
    path_list, url_list = [], [] # make unique lists of dem paths and downloadURLs

    print("Starting DEM Download Process...")
    all_datasets = ["Digital Elevation Model (DEM) 1 meter",
        "National Elevation Dataset (NED) 1/9 arc-second",
        "National Elevation Dataset (NED) 1/3 arc-second Current"]
    if resolution == "1 m":
        datasets = all_datasets
    elif resolution == "1/9 arc-second (~3 m)":
        datasets = all_datasets[1:]
    else:
        datasets = all_datasets[2:]

    for index, row in lhd_df.iterrows():
        if "dem_dir" not in lhd_df.columns or pd.isna(lhd_df.at[index, "dem_dir"]):
            lat = row['latitude']
            lon = row['longitude']
            bounding_dist = 2 * row.weir_length
            upper_lat, upper_lon = meters_to_latlon(lat, lon, bounding_dist, bounding_dist)
            lower_lat, lower_lon = meters_to_latlon(lat, lon, -1 * bounding_dist, -1 * bounding_dist)

            bbox = (lower_lon, lower_lat, upper_lon, upper_lat)
            # print(f"latitude = {lat}, longitude = {lon}")

            results = []
            for dataset in datasets:
                # define the API parameters
                params = {"bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
                          "datasets": dataset,
                          # "max": 10,
                          "prodFormats": ["GeoTIFF"],
                          "outputFormat": "JSON"}
                try:
                    response = requests.get(base_url, params=params)
                    response.raise_for_status() # not sure what this does
                    results = response.json().get("items", [])
                    if not results:
                        # print(f"No results found for {dataset} data.")
                        continue
                    else:
                        # print(f"Found {len(results)} result for {dataset} data.")
                        break
                except requests.RequestException as e:
                    print(f"Error occurred: {e}")
            if not results: # if there's no results for any (highly unlikely), it will skip over the dam and keep going
                continue
            dem = results[0]
            title = sanitize_filename(dem.get("title"))
            dem_subdir =   f"{dem_dir}/{title}"
            os.makedirs(dem_subdir, exist_ok=True) # if the directory already exists, it won't freak out
            dem_path = f"{dem_subdir}/{title}.tif"
            lhd_df.at[index, "dem_dir"] = dem_subdir # save the path to the specific dem folder to the DataFrame

            path_list.append(dem_path)
            url_list.append(dem.get("downloadURL"))

    unique_dems = {}
    for path, url in zip(path_list, url_list):
        if path not in unique_dems:
            unique_dems[path] = url

    for path, url in unique_dems.items():
        if not os.path.exists(path):
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            # print(f"Saved to {path}")
        else:
            print(f"File already exists at {path}")
    # lhd_df now has a column with the location of the dem associated with each dam
    return lhd_df

# """
# old test case
# """
# import pandas as pd
# database_csv = "C:/Users/ki87ujmn/Downloads/rathcelon-example/LHD_lat_long.csv"
# database_df = pd.read_excel(database_csv)
# output_loc = "./dem"
# new_df = download_dems(database_df, output_loc)
#
# new_df.to_csv("C:/Users/ki87ujmn/Downloads/test_dem.csv", index=False)
