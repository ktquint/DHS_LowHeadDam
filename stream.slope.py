# i like importing based on how long they are
import os
import re
import glob
import fiona
import zipfile
import requests
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point


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

# Source of data (initial excel file) - Remove later when df_slopes is available; use this for practice runs
file_path = "Low head Dam Info - Copy for python.xlsx"
df_slopes = pd.read_excel(file_path, usecols=['latitude', 'longitude', 'ID', 'LINKNO', 'gpkg'])

# Downloads the gpkg file later in the code
def download_gpkg(url, local_path):
    response = requests.get(url)
    with open(local_path, 'wb') as f:
        f.write(response.content)

def gpkg_download(df_slopes):
    # Place where gpkgs are downloaded
    download_dir = 'all_downloaded_gpkgs'
    os.makedirs(download_dir, exist_ok=True)

    #List of linknos, will just put the list into lhd
    linknos = pd.read_csv('list_of_linkno.csv')

    # Ensure the 'gpkg' column is of type object
    df_slopes['gpkg'] = df_slopes['gpkg'].astype(object)

    # looks through rows of the initial dataframe
    for row in df_slopes.itertuples():
        linkno = row.LINKNO # gets linkno value of the row

        if isinstance(linkno, int): # Checks if linkno is int
            found = False
            for column in linknos.columns: # checks thru all columns of linkno values csv
                local_gpkg_path = os.path.join(download_dir, f"{column}.gpkg") # path to store gpkg
                gpkg_url = f"http://geoglows-v2.s3-us-west-2.amazonaws.com/streams/{column}.gpkg" # where gpkg is downloaded from
                if linkno in linknos[column].values: # if linkno is in the column of the gpkg
                    if not os.path.exists(local_gpkg_path): # if wasn't already downloaded
                        download_gpkg(gpkg_url, local_gpkg_path) # downloads
                        df_slopes.at[row.Index, 'gpkg'] = f"{column}.gpkg" # adds gpkg name to the excel file
                        found = True
                        break
                    elif os.path.exists(local_gpkg_path): # if linkno is already downloaded
                        df_slopes.at[row.Index, 'gpkg'] = f"{column}.gpkg" # just adds gpkg name, no download
                        found = True
                        break
            if not found:
                df_slopes.at[row.Index, 'gpkg'] = "" # adds nothing if not found

    df_slopes.to_excel('output_w_gpkgs.xlsx', index=False) # converts to excel file

gpkg_download(df_slopes) #Works when I tested it, please check files
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

