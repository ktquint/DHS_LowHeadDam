import os
import re
import math
import requests
import rasterio
import pandas as pd
from datetime import datetime
from rasterio.merge import merge
from rasterio.io import MemoryFile
from rasterio.warp import calculate_default_transform, reproject, Resampling, transform_bounds


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
    dlat = int(dy) / R
    dlon = int(dx) / (R * math.cos(math.radians(lat0)))

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


def check_bbox_coverage(dem_path, bbox):
    """
    Check if all four corners of the bounding box are contained within the DEM
    and have valid elevation values (not NaN or nodata).

    Args:
        dem_path: Path to the merged DEM file
        bbox: Tuple of (min_lon, min_lat, max_lon, max_lat)

    Returns:
        bool: True if all corners are covered and have valid data, False otherwise
    """
    try:
        with rasterio.open(dem_path) as dem:
            dem_bounds = dem.bounds
            nodata = dem.nodata
            crs_needs_transform = dem.crs.to_string() != 'EPSG:4326'

            # Define the four corners (in EPSG:4326)
            corners = [
                (bbox[0], bbox[1]),  # bottom-left
                (bbox[2], bbox[1]),  # bottom-right
                (bbox[2], bbox[3]),  # top-right
                (bbox[0], bbox[3])  # top-left
            ]

            # Transform corners to raster CRS if needed
            if crs_needs_transform:
                xs, ys = zip(*corners)
                xs, ys = rasterio.warp.transform('EPSG:4326', dem.crs, xs, ys)
                corners = list(zip(xs, ys))

            # Check each corner
            for x, y in corners:
                # Bounds check
                if not (dem_bounds.left <= x <= dem_bounds.right and
                        dem_bounds.bottom <= y <= dem_bounds.top):
                    return False

                # Convert to pixel indices
                row, col = dem.index(x, y)
                if not (0 <= row < dem.height and 0 <= col < dem.width):
                    return False

                # Read elevation value
                value = dem.read(1)[row, col]
                if value == nodata or (isinstance(value, float) and math.isnan(value)):
                    return False

            return True

    except Exception as e:
        print(f"Error checking bbox coverage: {e}")
        return False


def download_dems(lhd_df, dem_dir, resolution):
    """
        this bad boy takes a DataFrame, finds the DEM associated with the lat/lon, records the DEM name in the DataFrame,
        downloads unique DEMs, and returns an updated DataFrame

        lhd_df: DataFrame with lat/lon on each low-head dam
        dem_dir: directory where DEM subdirectories will be located
        resolution: preferred resolution of the DEM
    """
    base_url = "https://tnmaccess.nationalmap.gov/api/v1/products"  # all products are found here

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

    # Ensure 'dem_dir' and 'dem_sanitized_names' column exists and is set to object (string-compatible)
    if "dem_dir" not in lhd_df.columns:
        lhd_df["dem_dir"] = ""
    lhd_df["dem_dir"] = lhd_df["dem_dir"].astype("object")
    if "dem_sanitized_names" not in lhd_df.columns:
        lhd_df["dem_sanitized_names"] = ""
    lhd_df["dem_sanitized_names"] = lhd_df["dem_sanitized_names"].astype("object")

    # List to store bbox coverage results
    bbox_coverage_results = []

    for index, row in lhd_df.iterrows():
        # let's check to see if this dam already has a DEM...
        lhd_id = row['ID']
        dem_subdir = os.path.join(dem_dir, f"{lhd_id}_DEM")
        os.makedirs(dem_subdir, exist_ok=True)  # if the directory already exists, it won't freak out
        dem_path = os.path.join(dem_subdir, f"{lhd_id}_MERGED_DEM.tif")

        # Check if DEM already exists and we already have the names stored
        if os.path.isfile(dem_path) and pd.notna(lhd_df.at[index, "dem_sanitized_names"]) and lhd_df.at[
            index, "dem_sanitized_names"].strip():
            print(f"DEM and names already exist for {lhd_id}, skipping")
            lhd_df.at[index, "dem_dir"] = dem_subdir
            continue

        # Get coordinates and make API calls to get DEM names
        print(f"Getting DEM info for {lhd_id}")
        lat = row['latitude']
        lon = row['longitude']

        weir_length = row.get("weir_length")
        if pd.isna(weir_length):
            print(f"Warning: weir_length is missing for dam {lhd_id}, skipping")
            continue

        bounding_dist = 2 * weir_length
        upper_lat, upper_lon = meters_to_latlon(lat, lon, bounding_dist, bounding_dist)
        lower_lat, lower_lon = meters_to_latlon(lat, lon, -1 * bounding_dist, -1 * bounding_dist)

        bbox = (lower_lon, lower_lat, upper_lon, upper_lat)

        # Variables to store the final results for this dam
        final_results = []
        final_titles = []
        final_download_urls = []
        dataset_used = None

        for dataset in datasets:
            # define the API parameters
            params = {"bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
                      "datasets": dataset,
                      # "max": 10,
                      "prodFormats": ["GeoTIFF"],
                      "outputFormat": "JSON"}
            try:
                response = requests.get(base_url, params=params)
                response.raise_for_status()
                results = response.json().get("items", [])
                """
                    what we does next depends on the resolution of the DEM, and the number of results
                """
                if len(results) == 0:
                    print(f"No results for {dataset}...\n Onto the next one!")
                    continue

                    # -------------------------- MULTIPLE 1-M DEMS --------------------------- #
                if dataset == "Digital Elevation Model (DEM) 1 meter" and len(results) > 1:
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
                    final_results = [entry[1] for entry in tile_to_item.values()]
                    final_titles = [sanitize_filename(dem.get("title", "")) for dem in final_results]
                    final_download_urls = [dem.get("downloadURL") for dem in final_results]

                elif dataset != "Digital Elevation Model (DEM) 1 meter" and len(results) > 1:
                    final_results = results
                    final_titles = [sanitize_filename(dem.get("title", "")) for dem in results]
                    final_download_urls = [dem.get("downloadURL") for dem in results]

                elif len(results) == 1:
                    final_results = results
                    final_titles = [sanitize_filename(results[0].get("title", ""))]
                    final_download_urls = [results[0].get("downloadURL")]

                # We found results, store the dataset used and break out of the loop
                dataset_used = dataset
                break

            except requests.RequestException as e:
                print(f"Error occurred: {e}")

        # Now store the DEM names ONCE per dam (outside the dataset loop)
        if final_titles:
            lhd_df.at[index, "dem_sanitized_names"] = final_titles
            print(f"Stored DEM names for {lhd_id}: {final_titles}")

        # Download only if merged DEM doesn't exist
        if final_results and not os.path.isfile(dem_path):
            print(f"Downloading DEM for {lhd_id}")

            if len(final_results) > 1:
                # Multiple DEMs - download and merge
                temp_paths = [os.path.join(dem_subdir, f"{title}.tif") for title in final_titles]
                print(temp_paths)
                for i in range(len(final_results)):
                    url = final_download_urls[i]
                    path = temp_paths[i]
                    with requests.get(url, stream=True) as r:
                        r.raise_for_status()
                        with open(path, "wb") as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                f.write(chunk)
                merge_dems(temp_paths, dem_path)
            else:
                # Single DEM - download directly
                with requests.get(final_download_urls[0], stream=True) as r:
                    r.raise_for_status()
                    with open(dem_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
        elif final_results:
            print(f"DEM file already exists for {lhd_id}, skipping download")

        # Check if we found any results first
        if not final_results:  # if there's no results for any dataset
            print(f"No DEM data found for {lhd_id}")
            continue
        # If we have results, check if DEM already exists
        elif os.path.isfile(dem_path):
            print(f"DEM file already exists for {lhd_id}, skipping download")

        # Check bounding box coverage (silently collect results)
        if os.path.isfile(dem_path):
            bbox_covered = check_bbox_coverage(dem_path, bbox)
            bbox_coverage_results.append({
                'lhd_id': lhd_id,
                'covered': bbox_covered,
                'dem_path': dem_path
            })

        lhd_df.at[index, "dem_dir"] = dem_subdir

    # Generate bbox coverage report at the end
    failed_coverage = [result for result in bbox_coverage_results if not result['covered']]

    if not failed_coverage:
        print("\nBBox coverage check complete, no issues found")
    else:
        print(f"\nBBox Coverage Report - {len(failed_coverage)} DEM(s) failed coverage check:")
        print("-" * 60)
        for failure in failed_coverage:
            print(f"Dam ID: {failure['lhd_id']} - INCOMPLETE coverage")
            print(f"  DEM Path: {failure['dem_path']}")
        print("-" * 60)

    # lhd_df now has a column with the location of the dem associated with each dam
    return lhd_df



