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
    dataset = "Lidar Point Cloud (LPC)"
    base_url = "https://tnmaccess.nationalmap.gov/api/v1/products"

    params = {
        "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
        "datasets": dataset,
        "max": 1,
        "outputFormat": "JSON"
    }

    response = requests.get(base_url, params=params)
    # print(response.text)
    lidar_info = response.json().get("items", [])

    if not lidar_info:
        print("No Lidar data found for the given coordinates.")
        return "No Lidar data found for the given coordinates."

    meta_url = lidar_info[0].get('metaUrl')
    if not meta_url:
        print("metaUrl key not found in the response.")
        return "metaUrl key not found in the response."

    response2 = requests.get(meta_url)
    # print(response2.text)
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
    historic_df = geoglows.data.retro_daily(comid)
    historic_df.index = pd.to_datetime(historic_df.index)

    if lat and lon is not None:
        try:
            date_range = get_dem_dates(lat, lon)
            print(date_range)
            subset_df = historic_df.loc[date_range[0]:date_range[1]]
            Q = np.median(subset_df[comid])
        except IndexError:
            date_range = get_dem_dates(lat, lon)
            raise ValueError(f"No data available for {date_range}")
    else:
        Q = np.median(historic_df[comid])

    return Q

def add_known_baseflow(lhd_df):
    lhd_df['known_baseflow'] = ""
    for index, row in lhd_df.iterrows():
        linkno = row.LINKNO
        lat = row.latitude
        lon = row.longitude
        dem_streamflow = get_streamflow(linkno, lat, lon)
        lhd_df.at[index, "known_baseflow"] = dem_streamflow
        print(f'index: {index}')
        print(f'known baseflow: {dem_streamflow}')
    return lhd_df


def add_dem_dates(lhd_df):
    lhd_df['dem_start'] = ""
    lhd_df['dem_end'] = ""
    for index, row in lhd_df.iterrows():
        lat = row.latitude
        lon = row.longitude
        lhd_df.at[index, "dem_start"] = get_dem_dates(lat, lon)[0]
        lhd_df.at[index, "dem_end"] = get_dem_dates(lat, lon)[1]
    return lhd_df
