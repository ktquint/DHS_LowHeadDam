import re
import io
import requests
import pandas as pd
import hydroinformatics as hi


gen_api_key = 'AIzaSyC4BXXMQ9KIodnLnThFi5Iv4y1fDR4U1II'

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
            return None

        data = response.json()
        lidar_info = data.get("items", [])
        if not lidar_info:
            print("No Lidar data found for the given coordinates.")
            return None

        meta_url = lidar_info[0].get('metaUrl')
        if not meta_url:
            print("metaUrl key not found in the response.")
            return None

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
            return None

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None

    except ValueError as ve:
        print(f"JSON decode error: {ve}")
        # noinspection PyUnboundLocalVariable
        print("Raw response text:", response.text[:300])
        return None


def estimate_dem_baseflow(stream_reach):
    """
        finds baseflow for a dem given latitude and longitude
    """
    # extract the lat and lon
    lat = stream_reach.latitude
    lon = stream_reach.longitude

    # get the date range of the lidar data
    dem_dates = get_dem_dates(lat, lon)
    # use the date range to estimate the baseflow
    if not dem_dates:   # if no dates given, just use the median flow
        dem_baseflow = stream_reach.get_median_flow()
    else:               # if there are dates, use them lol
        dem_baseflow = stream_reach.get_median_flow_in_range(dem_dates[0], dem_dates[1])

    return dem_baseflow
