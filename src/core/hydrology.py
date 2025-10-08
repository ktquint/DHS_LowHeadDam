import requests
import pandas as pd
import io
nwm_api_key = "AIzaSyC4BXXMQ9KIodnLnThFi5Iv4y1fDR4U1II"

def get_usgs_latlon(site_no):
    site_no = str(site_no).zfill(8)  # pad with leading zeros
    url = f"https://waterservices.usgs.gov/nwis/site/?site={site_no}&format=rdb"
    response = requests.get(url)

    lines = response.text.splitlines()

    # Remove all comment lines
    non_comment_lines = [line for line in lines if not line.startswith("#") and line.strip() != ""]

    # The format is:
    # 0: header line with column names
    # 1: field length descriptor line (skip this)
    # 2.: data lines

    if len(non_comment_lines) < 3:
        raise ValueError(f"No valid data found for site {site_no}")

    header_line = non_comment_lines[0]
    data_lines = non_comment_lines[2:]  # skip the field descriptor line (index 1)

    # Join header and data lines to make CSV text
    csv_text = "\n".join([header_line] + data_lines)

    df = pd.read_csv(io.StringIO(csv_text), sep="\t")

    # Check if latitude and longitude columns exist and not empty
    if df.empty or 'dec_lat_va' not in df.columns or 'dec_long_va' not in df.columns:
        raise ValueError(f"Latitude/Longitude columns missing for site {site_no}")

    lat = float(df['dec_lat_va'].iloc[0])
    lon = float(df['dec_long_va'].iloc[0])

    return lat, lon


def get_nwm_latlon(reach_id):
    url = f"https://nwm-api.ciroh.org/geometry?comids={reach_id}&key={nwm_api_key}"
    response = requests.get(url)

    if response.status_code != 200:
        raise ValueError(f"NWM API request failed with status {response.status_code}: {response.text}")

    data = response.json()[0]
    data_geom = data['geometry']
    coord_str = data_geom.replace("LINESTRING(", "").replace(")", "")
    # Split on commas
    coord_pairs = coord_str.split(", ")
    # Take first coordinate pair
    first_pair = coord_pairs[0]
    lon_str, lat_str = first_pair.split(" ")
    return float(lat_str), float(lon_str)  # lat, lon


streamflow_df = pd.read_csv("E:/LowHead_Dam_Streamflow.csv")
for index, row in streamflow_df.iterrows():
    comid = row['comid']
    source = row['source']
    if source == "USGS":
        print(source)
        comid = str(comid).zfill(8)
        print(comid)
        print(get_usgs_latlon(comid))
    elif source == "National Water Model":
        print(source)
        print(comid)
        print(get_nwm_latlon(comid))