import io
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv


env_path = Path(__file__).parent.parent / 'config' / '.env'
load_dotenv(dotenv_path=env_path)

# NWM API key is no longer used
# nwm_api_key = os.getenv("API_KEY")

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
    """
    This function is deprecated as the NWM API is no longer used.
    Latitude and longitude data should be loaded from the
    'nwm_daily_retrospective.parquet' file, which contains
    lat/lon coordinates for each feature_id.
    """
    raise NotImplementedError(
        "This function is deprecated. NWM API is no longer used. "
        "Load lat/lon data from the nwm_daily_retrospective.parquet file."
    )


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
        try:
            print(get_nwm_latlon(comid))
        except NotImplementedError as e:
            print(f"Skipping NWM comid {comid}: {e}")