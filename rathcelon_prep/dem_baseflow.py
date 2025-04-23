import requests

# Define the base URL

def get_dem_date(lhd_df):
    return lhd_df


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
