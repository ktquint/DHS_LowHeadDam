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


def get_reach_id(lat, lon, API_KEY=gen_api_key):
    r = requests.get(f"https://nwm-api.ciroh.org/geometry?lat={lat}&lon={lon}&output_format=csv&key={API_KEY}")
    # Check for successful response (HTTP status code 200)
    if r.status_code == 200:
        # Convert API response to pandas DataFrame
        df = pd.read_csv(io.StringIO(r.text))
        # Extract first (and only) reach ID from the response
        # print(df['station_id'].values)
        reach_id = df['station_id'].values[0]
        return reach_id
    else:
        # Raise error if API request fails
        raise requests.exceptions.HTTPError(r.text)


def add_known_baseflow(lhd_df, hydrology):
    """
    Adds known baseflow estimates to the dataframe for each dam,
    based on DEM LiDAR survey dates and hydrology source.
    """
    if 'known_baseflow' not in lhd_df.columns:
        lhd_df['known_baseflow'] = None

    if hydrology != "GEOGLOWS" and 'reach_id' not in lhd_df.columns:
        lhd_df['reach_id'] = None

    for row in lhd_df.itertuples(index=True):
        index = row.Index
        # skip rows with known base flows
        if pd.notnull(row.known_baseflow):
            continue

        lat = row.latitude
        lon = row.longitude
        date_range = get_dem_dates(lat, lon)

        if hydrology == "GEOGLOWS":
            comid = row.LINKNO
        else:
            comid = row.reach_id
            if pd.isnull(comid):
                comid = get_reach_id(lat, lon)
                lhd_df.at[index, 'reach_id'] = comid

        dem_baseflow = hi.get_streamflow(hydrology, comid, date_range)
        lhd_df.at[index, 'known_baseflow'] = dem_baseflow
        print(f'index: {index} | known baseflow: {dem_baseflow}')

    return lhd_df
