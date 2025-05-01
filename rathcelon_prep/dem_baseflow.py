import requests
import re


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
    lidar_info = response.json().get("items", [])

    if not lidar_info:
        return "No Lidar data found for the given coordinates."

    meta_url = lidar_info[0].get('metaUrl')
    if not meta_url:
        return "metaUrl key not found in the response."

    response2 = requests.get(meta_url)
    html_content = response2.text

    match_start = re.search(r'<dt>Start Date</dt>\s*<dd>(.*?)</dd>', html_content, re.IGNORECASE)
    match_end = re.search(r'<dt>End Date</dt>\s*<dd>(.*?)</dd>', html_content, re.IGNORECASE)

    if match_start and match_end:
        start_date_value = match_start.group(1).strip()
        end_date_value = match_end.group(1).strip()
        return f"Start Date: {start_date_value}, End Date: {end_date_value}"
    else:
        return "Date parameters not found."

get_dem_dates(36.12085558,-95.98829985)




def get_dem_discharge(lhd_df):
    """
    lhd_df: must have lat/lon column
    from lat/lon we get DEM data (metadata)
    lat/lon will also give us river_id??
    metadata will give us data
    """
    lhd_df["dem_baseflow"] = ""  # initialize column
    base_url = "https://geoglows.ecmwf.int/api/v2/retrospectivedaily"


    for index, row in lhd_df.iterrows():
        # lat = row.latitude
        # lon = row.longitude

        # eventually river_id will come from lat/lon
        river_id = row.river_id
        # dem_date will come from the dem metadata
        dem_date = '20170101'

        # update parameters for the specific dam/dem
        river_url = f"{base_url}/{river_id}"
        params = {
            # 'river_id': river_id, # bruh, it doesn't recognize river_id here
            'format': 'json',
            'start_date': dem_date,
            'end_date': dem_date,
            'bias_corrected': True
        }

        # Make the request
        response = requests.get(river_url, params=params)

        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()
            baseflow = data[river_id]
            lhd_df.at[index, 'dem_baseflow'] = baseflow
        else:
            print(f"Error: {response.status_code}")
            print(response.json())  # Print the error message for debugging

    return lhd_df
