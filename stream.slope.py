# import all the packages we'll need
import glob
import os
import re
import requests
import pandas as pd
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


def search_and_download_gdb(df, output_folder):
    """
    - find the geo-database that contains the hydrography around a streamgage
    **right now the USGS api returns all the surrounding ones... so for now I'll grab everything
    """
    # Make a set of unique geodatabases
    gdbs_unique = set()
    for row in df.itertuples():
        lat = row.latitude
        lon = row.longitude
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
                return

            # clean out list of geodatabases
            gdb_list = []
            for item in results:
                # only add geodatabases
                if item['format'] == 'FileGDB, NHDPlus HR Rasters':
                    # save a list of geodatabases associated with this dam's lat-lon
                    gdb_list.append(item)
                    # update the unique set of geodatabases
                    gdbs_unique.update(gdb_list)

            # make the output folder if it doesn't already exist
            os.makedirs(output_folder, exist_ok=True)

            for item in gdbs_unique:
                title = item.get("title", "Unnamed")
                sanitized_title = sanitize_filename(title)  # sanitize the file name
                download_url = item.get("downloadURL")

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


def merge_tables(df, buffer_distance=1/3600):
    """
    - make buffer around lat/lon
    - look through all the gdbs and find the one that contains the right stream
    - load the vaa table and merge attributes
    - return the slope value
    """
    # df is dataframe, is input from earlier code
    # Initialize a list to store the max slope values
    max_slope_values = []
    # create a point using gage latitude and longitude

    for index, row in df.iterrows():
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
    df['Slope'] = max_slope_values
    df.to_excel('output.xlsx', index=False)
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


def slope_from_lat_lon (streamdata):

    # Read the Excel file into a DataFrame
    df = pd.read_excel(streamdata)

    # Select specific columns
    df_clean = df[['ID', 'latitude', 'longitude', 'Width (ft)']]
    df_gdbs = search_and_download_gdb(df_clean, './gdbs')
    df_slopes = merge_tables(df_gdbs, './gdbs')
    output_excel = streamdata[:-4] + '_slopes.xlsx'
    df_slopes.to_excel(output_excel, index=False)
    # Display the selected columns: print(selected_columns)

''' def slope_lat_long (streamdata.xlsx)
        read excel to df
        download_gdb(df)
            return df + list of gdb, loc of unique gdb
        merge_tables (df2.0 , loc)
            return df 3.0
        save df to csu or excel'''

# start of Eliana's code
import os
import warnings
import requests
import geopandas as gpd
from io import BytesIO
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

df_slopes = pd.read_excel("practice_download_gpkg.xlsx", usecols = ['latitude', 'longitude','ID', 'LINKNO','gpkg'])

def check_linkno_in_gpkg(url, linkno_value):
    logging.info(f"Checking URL: {url}")
    # Suppress the RuntimeWarning
    warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*non conformant file extension.*")

    # Fetch a small portion of the file to check for LINKNO value
    response = requests.get(url, stream=True)
    response.raise_for_status()

    # Read the file into a GeoDataFrame
    with BytesIO(response.content) as f:
        gdf = gpd.read_file(f)

    # Check if the LINKNO value exists in the GeoDataFrame
    if linkno_value in gdf['LINKNO'].values:
        return True
    return False


def download_gpkg(url, output_path):
    response = requests.get(url)
    response.raise_for_status()

    with open(output_path, 'wb') as f:
        f.write(response.content)


# Example usage
base_url = 'https://geoglows-v2.s3-us-west-2.amazonaws.com/streams/streams_'
start_index = 101
output_directory = 'C:/Users/Owner/Python Practice/gpkg_files_practice'  # Change this to your desired directory

# Ensure the .gpkg column is of type object
df_slopes['gpkg'] = df_slopes['gpkg'].astype(object)

try:
    for row in df_slopes.itertuples():
        if pd.isna(row.gpkg):  # Check if the .gpkg column is empty
            linkno_value = row.LINKNO
            for i in range(start_index, start_index + 704):  # Adjust the range as needed
                url = f"{base_url}{i}.gpkg"
                try:
                    if check_linkno_in_gpkg(url, linkno_value):
                        output_path = os.path.join(output_directory, f"streams_{i}.gpkg")
                        download_gpkg(url, output_path)
                        logging.info(f"Downloaded {output_path} containing LINKNO {linkno_value}")

                        # Update the .gpkg column with the name of the downloaded file
                        df_slopes.at[row.Index, 'gpkg'] = f"streams_{i}.gpkg"
                        break  # Exit the inner loop once the file is found and downloaded
                except requests.HTTPError as e:
                    logging.error(f"Failed to access {url}: {e}")
                except Exception as e:
                    logging.error(f"An error occurred with {url}: {e}")
except KeyboardInterrupt:
    logging.info("Script interrupted by user. Exiting gracefully...")

# Save the updated DataFrame to a file if needed
df_slopes.to_excel('updated_slopes.xlsx', index=False)
# End of Eliana's code

"""
Old Test Case:
"""
# gage_info = str(input("Enter gage info: "))
#
# latitude, longitude, station_id = extract_lat_lon_and_station_id(gage_info)
# output_folder = './' + station_id
#
# print(f'Station ID: {station_id}')
# print(f'Latitude: {latitude}')
# print(f'Longitude: {longitude}')
#
# stream_slope = slope_from_lat_lon(latitude, longitude, output_folder)
# print(f'The stream slope at USGS Station {station_id} is {stream_slope}')

