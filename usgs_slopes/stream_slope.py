# i like importing based on how long they are
import os
import re
import glob
import fiona
import zipfile
import requests
import pandas as pd
import geopandas as gpd
from requests.structures import CaseInsensitiveDict
from shapely.geometry import Point


def sanitize_filename(filename):
    """replace or remove invalid characters from a file name."""
    invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    for char in invalid_chars:
        filename = filename.replace(char, "_")  # Replace invalid characters with '_'
    return filename

def search_and_download_gdb(lhd_df, output_folder):
    """
    - find the geo-database that contains the hydrography around a streamgage
    **right now the USGS api returns all the surrounding ones... so for now I'll grab everything
    """
    # Make a set of unique geodatabases
    gdbs_unique = set()

    # Add a column to the lhd data frame to store the sanitized titles
    lhd_df['Sanitized Titles'] = None

    for index, row in lhd_df.iterrows():
        lat = row['latitude']
        lon = row['longitude']
        # Skip rows with missing latitude or longitude
        if pd.isna(lat) or pd.isna(lon):
            continue

        bbox = (lon - 0.0003, lat - 0.0003, lon + 0.0003, lat + 0.0003)

        product = "National Hydrography Dataset Plus High Resolution (NHDPlus HR)"
        usgs_api = "https://tnmaccess.nationalmap.gov/api/v1/products"

        params = {
            "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
            "datasets": product,
            "max": 10,  # number of results to return
            "outputFormat": "JSON"
        }

        try:
            # query the API
            response = requests.get(usgs_api, params=params)
            response.raise_for_status()

            # parse the results
            results = response.json().get("items", [])
            if not results:
                print(f"No results found for {product} data.")
                continue

            # clean out list of geodatabases
            gdb_list = []
            for item in results:
                # only add geodatabases
                if item['format'] == 'FileGDB, NHDPlus HR Rasters':
                    # save a list of geodatabases associated with this dam's lat-lon
                    gdb_list.append(item)


            # Get the names of the gdb_list and sanitize them
            sanitized_titles = [sanitize_filename(item.get("title", "Unnamed")) for item in gdb_list]
            # Add the sanitized titles to the DataFrame
            lhd_df.at[index, 'Sanitized Titles'] = str(sanitized_titles)


            # make the output folder if it doesn't already exist
            os.makedirs(output_folder, exist_ok=True)

            for item in gdb_list:
                unique_downloadurl = dict(item).get("downloadURL")
                if unique_downloadurl not in gdbs_unique:
                    download_url = unique_downloadurl
                    gdbs_unique.add(download_url)
                else:
                    continue

                #Get the names of the gdb_list
                title = dict(item).get("title", "Unnamed")
                sanitized_title = sanitize_filename(title)  # sanitize the file name

                if download_url:
                    local_zip = os.path.join(output_folder, f"{sanitized_title}.zip")
                    print(f"Retrieving {sanitized_title}...")

                    with requests.get(download_url, stream=True) as r:
                        r.raise_for_status()
                        with open(local_zip, "wb") as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                f.write(chunk)
                    # print(f"Saved to {local_zip_path}")

                    # unzip the zip file
                    with zipfile.ZipFile(local_zip, 'r') as zip_ref:
                        zip_ref.extractall(output_folder)

                    # let's remove the zip file after extraction
                    os.remove(local_zip)
                else:
                    print(f"No download URL for {title}. womp womp")

        except requests.RequestException as e:
            print(e)


def merge_tables(lhd_df, buffer_distance=1/3600):
    """
    - make buffer around lat/lon
    - look through all the gdbs and find the one that contains the right stream
    - load the vaa table and merge attributes
    - return the slope value
    """
    # lhd_df is dataframe, is input from earlier code
    # Initialize a list to store the max slope values
    max_slope_values = []
    # create a point using gage latitude and longitude

    for index, row in lhd_df.iterrows():
        # get lat, lon from dataframe
        lat = row['latitude']
        lon = row['longitude']

        # create a point using gage latitude and longitude
        point = Point(lon, lat)
        # create a buffer around the point
        buffer = point.buffer(buffer_distance)
        # possibly change: find a way to convert the row of geodatabase to actual gdb_files
        gdb_files = row['geodatabase']
        # initialize nhd_flowline and vaa_table
        nhd_flowline = gpd.GeoDataFrame()
        vaa_table = gpd.GeoDataFrame()
        selected_flowlines = []
        slope_value = []

        for gdb in gdb_files:
            huc = gdb.split('_')[2]
            print("Reading in NHDFlowline hydrography for Hydrologic Unit – " + huc)
            nhd_flowline = gpd.read_file(gdb, layer='NHDFlowline', engine='fiona')
            # maybe try finding flowline based on distance...
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

        # Add the max slope value to the list
        if slope_value:
            max_slope_values.append(max(slope_value))
        else:
            max_slope_values.append(None)
    # Add the max slope values to the DataFrame
    lhd_df['Slope'] = max_slope_values
    lhd_df.to_excel('output.xlsx', index=False)
    return # a dataframe with a slope column


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


def slope_from_lat_lon (lhd_xlsx):

    # Read the Excel file into a DataFrame
    lhd_df = pd.read_excel(lhd_xlsx)

    # Select specific columns
    lhd_df_clean = lhd_df[['ID', 'latitude', 'longitude', 'Width (ft)']]
    os.makedirs('./gdbs', exist_ok=True)
    lhd_df_gdbs = search_and_download_gdb(lhd_df_clean, './gdbs')
    lhd_df_slopes = merge_tables(lhd_df_gdbs)
    lhd_df_slopes.to_excel(lhd_xlsx, index=False)
    # Display the selected columns: print(selected_columns)

''' def slope_lat_long (streamdata.xlsx)
        read excel to df
        download_gdb(df)
            return df + list of gdb, loc of unique gdb
        merge_tables (df2.0 , loc)
            return df 3.0
        save df to csu or excel'''


"""
Old Test Case:
"""
'''gage_info = str(input("Enter gage info: "))

latitude, longitude, station_id = extract_lat_lon_and_station_id(gage_info)
output_folder = './' + station_id

print(f'Station ID: {station_id}')
print(f'Latitude: {latitude}')
print(f'Longitude: {longitude}')

stream_slope = slope_from_lat_lon(lhd_database_xlsx)
print(f'The stream slope at USGS Station {station_id} is {stream_slope}')'''

''''###
Test Case
###'''
# lhd_csv = "C:/Users/pgordi/Downloads/LHD stream-slope test cases/LHD database test case 2.1 - duplicate downloads.csv"
# lhd_df = pd.read_csv(lhd_csv)
# output_folder = "C:/Users/pgordi/Downloads/LHD downloads"
# search_and_download_gdb(lhd_df, output_folder)
# lhd_df.to_csv("C:/Users/pgordi/Downloads/LHD stream-slope test cases/LHD database test case 2.1 - duplicate downloads.csv", index=False)