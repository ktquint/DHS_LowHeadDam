import os
import requests


def sanitize_filename(filename):
    """
    some files have '/' in their name like "1/9 arc-second," so we'll fix it
    """
    invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    for char in invalid_chars:
        filename = filename.replace(char, "-")  # Replace invalid characters with '_'
    return filename


def download_dems(lhd_df, dem_dir):
    """
    this bad boy takes a DataFrame, finds the DEM associated with the lat/lon, records the DEM name in the DataFrame,
    downloads unique DEMs, and returns an updated DataFrame

    lhd_df: DataFrame with lat/lon on each low-head dam
    dem_dir: directory where DEM subdirectories will be located
    """
    lhd_df["dem_dir"] = "" # initialize column with dem locations
    base_url = "https://tnmaccess.nationalmap.gov/api/v1/products" # all products are found here
    path_list, url_list = [], [] # make unique lists of dem paths and downloadURLs

    print("Starting DEM Download Process...")

    for index, row in lhd_df.iterrows():
        lat = row.latitude
        lon = row.longitude
        bbox = (lon - 0.001, lat - 0.001, lon + 0.001, lat + 0.001)
        # print(f"latitude = {lat}, longitude = {lon}")
        datasets = [# "Digital Elevation Model (DEM) 1 meter",
                    "National Elevation Dataset (NED) 1/9 arc-second",
                    "National Elevation Dataset (NED) 1/3 arc-second Current"]
        results = []
        for dataset in datasets:
            # define the API parameters
            params = {"bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
                      "datasets": dataset,
                      "max": 1,
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
        url_list.append(dem.get('downloadURL'))

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

"""
old test case
"""
# database_csv = "C:/Users/ki87ujmn/Downloads/rathcelon-example/LHD_lat_long.csv"
# database_df = pd.read_csv(database_csv)
# output_loc = "./dem"
# new_df = download_dems(database_df, output_loc)
#
# new_df.to_csv("C:/Users/ki87ujmn/Downloads/test_dem.csv", index=False)
