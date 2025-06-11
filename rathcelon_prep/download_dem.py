import os
import re
import math
import requests
import rasterio
from datetime import datetime
from rasterio.merge import merge
from rasterio.io import MemoryFile
from rasterio.warp import calculate_default_transform, reproject, Resampling


def extract_date(item):
    return item.get("dateCreated") or item.get("publishedDate") or ""

def extract_tile_id(title):
    match = re.search(r'x\d+y\d+', title)
    return match.group(0) if match else None

def sanitize_filename(filename):
    """
        some files have '/' in their name like "1/3 arc-second," so we'll fix it
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
    dem_dir = os.path.dirname(dem_files[0])
    output_loc = os.path.join(dem_dir, output_filename)

    mosaic_inputs = []
    memfiles = []
    target_crs = None

    for fp in dem_files:
        src = rasterio.open(fp)
        if target_crs is None:
            target_crs = src.crs

        if src.crs != target_crs:
            transform, width, height = calculate_default_transform(
                src.crs, target_crs, src.width, src.height, *src.bounds)

            kwargs = src.meta.copy()
            kwargs.update({
                'crs': target_crs,
                'transform': transform,
                'width': width,
                'height': height
            })

            memfile = MemoryFile()
            memfiles.append(memfile)  # Keep reference alive
            dst = memfile.open(**kwargs)

            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.nearest)

            dst.close()
            dst = memfile.open()
            mosaic_inputs.append(dst)
        else:
            mosaic_inputs.append(src)

    mosaic, out_transform = merge(mosaic_inputs)

    out_meta = mosaic_inputs[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_transform,
        "crs": target_crs
    })

    with rasterio.open(output_loc, "w", **out_meta) as dest:
        dest.write(mosaic)

    # Clean up
    for src in mosaic_inputs:
        src.close()
    for memfile in memfiles:
        memfile.close()

    # Delete original files
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
        # let's check to see if this dam already has a DEM...
        lhd_id = row['ID']
        dem_subdir = os.path.join(dem_dir, f"{lhd_id}_DEM")
        os.makedirs(dem_subdir, exist_ok=True)  # if the directory already exists, it won't freak out
        dem_path = os.path.join(dem_subdir, f"{lhd_id}_MERGED_DEM.tif")

        if not os.path.isfile(dem_path):
            print(f"Downloading DEM for {lhd_id}")
            lat = row['latitude']
            lon = row['longitude']
            bounding_dist = 2 * row.weir_length
            upper_lat, upper_lon = meters_to_latlon(lat, lon, bounding_dist, bounding_dist)
            lower_lat, lower_lon = meters_to_latlon(lat, lon, -1 * bounding_dist, -1 * bounding_dist)

            bbox = (lower_lon, lower_lat, upper_lon, upper_lat)

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
                    """
                        what we does next depends on the resolution of the DEM, and the number of results
                    """
                    if len(results) == 0:
                        print(f"No results for {dataset}...\n Onto the next one!")
                        continue

                    # -------------------------- MULTIPLE 1-M DEMS --------------------------- #
                    elif dataset == "Digital Elevation Model (DEM) 1 meter" and len(results) > 1:
                        print(results)
                        # Filter to get only the most recent DEM per tile
                        filtered = []
                        for item in results:
                            title = item.get("title", "")
                            if extract_tile_id(title) and extract_date(item):
                                filtered.append(item)

                        # Build dict of most recent item per tile
                        tile_to_item = {}
                        for item in filtered:
                            title = item.get("title", "")
                            tile_id = extract_tile_id(title)
                            date_str = extract_date(item)
                            try:
                                date_obj = datetime.fromisoformat(date_str)
                            except ValueError:
                                continue  # skip if date format is wrong

                            if tile_id not in tile_to_item or date_obj > tile_to_item[tile_id][0]:
                                tile_to_item[tile_id] = (date_obj, item)

                        # Extract only the most recent item per tile
                        results = [entry[1] for entry in tile_to_item.values()]
                        titles = [sanitize_filename(dem.get("title", "")) for dem in results]
                        download_urls = [dem.get("downloadURL") for dem in results]

                        temp_paths = [os.path.join(dem_subdir, f"{title}.tif") for title in titles]
                        print(temp_paths)
                        for i in range(len(results)):
                            url = download_urls[i]
                            path = temp_paths[i]
                            with requests.get(url, stream=True) as r:
                                r.raise_for_status()
                                with open(path, "wb") as f:
                                    for chunk in r.iter_content(chunk_size=8192):
                                        f.write(chunk)
                        merge_dems(temp_paths, dem_path)

                    # ------------------ MULTIPLE 1/3 OR 1/9 ARC-SECOND DEMS ----------------- #
                    elif dataset != "Digital Elevation Model (DEM) 1 meter" and len(results) > 1:
                        titles = [sanitize_filename(dem.get("title", "")) for dem in results]
                        download_urls = [dem.get("downloadURL") for dem in results]
                        temp_paths = [os.path.join(dem_subdir, f"{title}.tif") for title in titles]
                        for i in range(len(results)):
                            url = download_urls[i]
                            path = temp_paths[i]
                            with requests.get(url, stream=True) as r:
                                r.raise_for_status()
                                with open(path, "wb") as f:
                                    for chunk in r.iter_content(chunk_size=8192):
                                        f.write(chunk)
                        merge_dems(temp_paths, dem_path)

                    # ---- SINGULAR DEMS ---- #
                    elif len(results) == 1:
                        download_urls = [dem.get("downloadURL") for dem in results]
                        with requests.get(download_urls[0]) as r:
                            r.raise_for_status()
                            with open(dem_path, "wb") as f:
                                for chunk in r.iter_content(chunk_size=8192):
                                    f.write(chunk)

                except requests.RequestException as e:
                    print(f"Error occurred: {e}")
            if not results: # if there's no results for any (highly unlikely), it will skip over the dam and keep going
                continue

        print(dem_subdir)
        lhd_df.at[index, "dem_dir"] = dem_subdir
    # lhd_df now has a column with the location of the dem associated with each dam
    return lhd_df
