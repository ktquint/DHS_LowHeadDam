import re
import geoglows
import requests
import numpy as np
import pandas as pd


def get_dem_dates(lat, lon):
    """
    Use lat/lon to get Lidar data used to make the DEM.
    Check the date the Lidar was taken.
    """
    bbox = (lon - 0.001, lat - 0.001, lon + 0.001, lat + 0.001)
    base_url = "https://tnmaccess.nationalmap.gov/api/v1/products"

    params = {
        "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
        "datasets": "Lidar Point Cloud (LPC)",
        "max": 1,
        "outputFormat": "JSON"
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()

        if 'application/json' not in response.headers.get("Content-Type", ""):
            print("Response not in JSON format:", response.text)
            return "Invalid format"

        data = response.json()
        lidar_info = data.get("items", [])
        if not lidar_info:
            print("No Lidar data found for the given coordinates.")
            return "No Lidar data found for the given coordinates."

        meta_url = lidar_info[0].get('metaUrl')
        if not meta_url:
            print("metaUrl key not found in the response.")
            return "metaUrl key not found in the response."

        response2 = requests.get(meta_url)
        html_content = response2.text

        match_start = re.search(r'<dt>Start Date</dt>\s*<dd>(.*?)</dd>', html_content, re.IGNORECASE)
        match_end = re.search(r'<dt>End Date</dt>\s*<dd>(.*?)</dd>', html_content, re.IGNORECASE)

        if match_start and match_end:
            start_date_value = match_start.group(1).strip()
            end_date_value = match_end.group(1).strip()
            return [start_date_value, end_date_value]
        else:
            print("Date parameters not found.")
            return "Date parameters not found."

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return "Request failed"

    except ValueError as ve:
        print(f"JSON decode error: {ve}")
        # noinspection PyUnboundLocalVariable
        print("Raw response text:", response.text[:300])
        return "JSON decode error"


def get_streamflow(comid, lat=None, lon=None):
    """
    comid needs to be an int
    date needs to be in the format %Y-%m-%d

    returns average streamflow—for the entire record if no lat-long is given, else it's the average from the dates the lidar was taken
    """
    try:
        comid = int(comid)
    except ValueError:
        raise ValueError("comid needs to be an int")

    # this is all the data for the comid
    historic_df = geoglows.data.retro_daily(comid, bias_corrected=True)
    historic_df.index = pd.to_datetime(historic_df.index)

    if lat and lon is not None:
        try:
            date_range = get_dem_dates(lat, lon)
            # print(date_range)
            if len(date_range) == 2:
                subset_df = historic_df.loc[date_range[0]:date_range[1]]
                Q = np.median(subset_df[comid])
            else: # if it returns an error statement, just return the historic median
                Q = np.median(historic_df[comid])
        except IndexError:
            date_range = get_dem_dates(lat, lon)
            raise ValueError(f"No data available for {date_range}")
    else:
        Q = np.median(historic_df[comid])
    return Q


def add_known_baseflow(lhd_df, hydrology):
    if 'known_baseflow' not in lhd_df.columns:
        lhd_df['known_baseflow'] = None

    if hydrology == "GEOGLOWS":
        for index, row in lhd_df.iterrows():
            # skip rows that already have known base flows
            if pd.notnull(row['known_baseflow']):
                continue

            linkno = row["LINKNO"]
            lat = row["latitude"]
            lon = row["longitude"]

            dem_streamflow = get_streamflow(linkno, lat, lon)
            lhd_df.at[index, "known_baseflow"] = dem_streamflow
            print(f'index: {index}')
            print(f'known baseflow: {dem_streamflow}')
    else:
        print("NWM is not ready... lol")
    return lhd_df


def add_dem_dates(lhd_df):
    lhd_df['dem_start'] = None
    lhd_df['dem_end'] = None
    for index, row in lhd_df.iterrows():
        lat = row.latitude
        lon = row.longitude
        date_range = get_dem_dates(lat, lon)
        if isinstance(date_range, list) and len(date_range) == 2:
            lhd_df.at[index, "dem_start"] = date_range[0]
            lhd_df.at[index, "dem_end"] = date_range[1]
    return lhd_df
