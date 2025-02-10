# import all the packages we'll need
import glob
import os
import re
import requests
import geopandas as gpd
import fiona
from shapely.geometry import Point
import zipfile

def sanitize_filename(filename):
    """replace or remove invalid characters from a file name."""
    invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    for char in invalid_chars:
        filename = filename.replace(char, "_")  # Replace invalid characters with '_'
    return filename


def search_and_download_gdb(lat, long, output_folder):
    """
    - find the geo-database that contains the hydrography around a streamgage
    **right now the USGS api returns all the ones around it... so for now I'll grab everything
    """
    bbox = (long - 0.0003, lat - 0.0003, long + 0.0003, lat + 0.0003)

    product = "National Hydrography Dataset Plus High Resolution (NHDPlus HR)"
    base_url = "https://tnmaccess.nationalmap.gov/api/v1/products"

    params = {
        "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
        "datasets": product,
        "max": 10,  # number of results to return
        "outputFormat": "JSON"
    }

    try:
        # query the API
        response = requests.get(base_url, params=params)
        response.raise_for_status()

        # parse the results
        results = response.json().get("items", [])
        if not results:
            print(f"No results found for {product} data.")
            return

        gdbs = []
        for item in results:
            if item['format'] == 'FileGDB, NHDPlus HR Rasters':
                gdbs.append(item)

        os.makedirs(output_folder, exist_ok=True)

        for item in gdbs:
            title = item.get("title", "Unnamed")
            sanitized_title = sanitize_filename(title)  # sanitize the file name
            download_url = item.get("downloadURL")

            if download_url:
                local_zip_path = os.path.join(output_folder, f"{sanitized_title}.zip")
                print(f"Retrieving {sanitized_title}...")

                with requests.get(download_url, stream=True) as r:
                    r.raise_for_status()
                    with open(local_zip_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                # print(f"Saved to {local_zip_path}")

                # unzip the zip file
                with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
                    zip_ref.extractall(output_folder)

                # let's remove the zip file after extraction
                os.remove(local_zip_path)
            else:
                print(f"No download URL for {title}. womp womp")

    except requests.RequestException as e:
        print(e)


def merge_tables(gdb_files, lat, lon, buffer_distance=1/3600):
    """
    - make buffer around lat/long
    - look through all the gdbs and find the one that contains the right stream
    - load the vaa table and merge attributes
    - return the slope value
    """
    # create a point using gage latitude and longitude
    point = Point(lon, lat)
    # create a buffer around the point
    buffer = point.buffer(buffer_distance)
    # initialize nhd_flowline and vaa_table
    nhd_flowline = gpd.GeoDataFrame()
    vaa_table = gpd.GeoDataFrame()

    for gdb in gdb_files:
        huc = gdb.split('_')[2]
        print("Reading in NHDFlowline hydrography for Hydrologic Unit – " + huc)
        nhd_flowline = gpd.read_file(gdb, layer='NHDFlowline', engine='fiona')
        selected_flowlines = nhd_flowline[nhd_flowline.intersects(buffer)]
        # reset the buffer distance
        buffer_distance = 1/3600

        counter = 0
        while len(selected_flowlines) != 1:
            counter += 1
            if counter > 17:
                print("Your gage is not located in Hydrologic Unit – " + huc + '... onto the next one')
                break
            if selected_flowlines.empty:
                buffer_distance += .1/3600
                buffer = point.buffer(buffer_distance)
                selected_flowlines = nhd_flowline[nhd_flowline.intersects(buffer)]
            else:
                buffer_distance -= .1/3600
                buffer = point.buffer(buffer_distance)
                selected_flowlines = nhd_flowline[nhd_flowline.intersects(buffer)]

        if len(selected_flowlines) == 1:
            print("Reading in NHDPlusFlowlineVAA Table...")
            break

    print("Joining attributes based on NHDPlusID")
    # sometimes the fields aren't capitalized... so we'll have to check for that
    if 'NHDPlusID' not in nhd_flowline.columns:
        vaa_table = gpd.read_file(gdb, layer='NHDPlusFlowlineVAA', engine='fiona', columns=['nhdplusid', 'slope'])
        joined = selected_flowlines.merge(vaa_table, on='nhdplusid')
        slope_value = joined['slope'].tolist()
    else:
        vaa_table = gpd.read_file(gdb, layer='NHDPlusFlowlineVAA', engine='fiona', columns=['NHDPlusID', 'Slope'])
        joined = selected_flowlines.merge(vaa_table, on='NHDPlusID')
        slope_value = joined['Slope'].tolist()

    return max(slope_value)


def find_geo_files(folder_path):
    # find the directory that ends in .gdb
    gdb_files = glob.glob(os.path.join(folder_path, '**', '*.gdb'), recursive=True)
    if not gdb_files:
        gdb_files = glob.glob(os.path.join(folder_path, '**', '*.gpkg'), recursive=True)
    return reversed(gdb_files)


def extract_lat_lon_and_station_id(input_string):
    # Split the input string by spaces
    parts = input_string.split()

    # Initialize variables to store the floats and the integer
    coords = []
    station = None

    # Iterate through the parts and identify floats and the integer
    for part in parts:
        try:
            # Try to convert the part to a float
            float_value = float(part)
            # Check if the float value is actually an integer
            if '.' in part:
                coords.append(float_value)
            else:
                station = int(float_value)
        except ValueError:
            # If it's neither a float nor an integer, ignore it
            pass

    return coords[0], coords[1], str(station)


def slope_from_lat_lon(lat, lon, output_loc):
    search_and_download_gdb(lat, lon, output_loc)
    gdb_paths = find_geo_files(output_loc)
    slope = merge_tables(gdb_paths, lat, lon)
    return slope


gage_info = str(input("Enter gage info: "))

latitude, longitude, station_id = extract_lat_lon_and_station_id(gage_info)
output_folder = './' + station_id

print(f'Station ID: {station_id}')
print(f'Latitude: {latitude}')
print(f'Longitude: {longitude}')

stream_slope = slope_from_lat_lon(latitude, longitude, output_folder)
print(f'The stream slope at USGS Station {station_id} is {stream_slope}')

