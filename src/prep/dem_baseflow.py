import os
import requests
import tempfile
import laspy
import numpy as np
from scipy.spatial import cKDTree
from pyproj import Transformer
from datetime import datetime, timedelta

# global variables
gen_api_key = 'AIzaSyC4BXXMQ9KIodnLnThFi5Iv4y1fDR4U1II'
base_url = "https://tnmaccess.nationalmap.gov/api/v1/products"


def gpstime_to_date(gps_time: float) -> str:
    """
    Converts GPS time (seconds since 1980-01-06) to a date string 'yyyy-MM-DD'.

    Parameters:
        gps_time (float): GPS time in seconds.

    Returns:
        str: Date string in 'yyyy-MM-DD' format.
    """
    gps_epoch = datetime(1980, 1, 6)
    utc_time = gps_epoch + timedelta(seconds=gps_time)

    return utc_time.strftime('%Y-%m-%d')


def find_water_gpstime(lat, lon):
    bbox = (lon - 0.001, lat - 0.001, lon + 0.001, lat + 0.001)

    params = {"bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
              "datasets": "Lidar Point Cloud (LPC)",
              "max": 1, "outputFormat": "JSON" }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()

        data = response.json()
        items = data.get("items", [])
        if not items:
            print("No Lidar data found.")
            return None

        download_url = items[0].get("downloadURL")
        if not download_url:
            print("No download URL found.")
            return None

        print("Downloading LiDAR file...")
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "lidar_data")

            lidar_response = requests.get(download_url)
            lidar_response.raise_for_status()

            # Determine extension
            content_type = lidar_response.headers.get("Content-Type", "").lower()
            if "zip" in content_type or download_url.endswith(".zip"):
                file_path += ".zip"
            elif download_url.endswith(".laz"):
                file_path += ".laz"
            elif download_url.endswith(".las"):
                file_path += ".las"
            else:
                print("Unknown file type.")
                return None

            with open(file_path, 'wb') as f:
                f.write(lidar_response.content)

            if file_path.endswith(".zip"):
                import zipfile
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(tmpdir)
                laz_files = [f for f in os.listdir(tmpdir) if f.lower().endswith((".las", ".laz"))]
                if not laz_files:
                    print("No LAS/LAZ files found in ZIP.")
                    return None
                las_path = os.path.join(tmpdir, laz_files[0])
            else:
                las_path = file_path  # raw .las or .laz file

            print(f"Processing {os.path.basename(las_path)}...")
            las = laspy.read(las_path)

            x = las.x
            y = las.y

            # Get CRS and project the lat/lon
            crs = las.header.parse_crs()
            if crs is None:
                print("CRS not found in LAS header.")
                return None

            crs_auth = crs.sub_crs_list[0].to_authority()
            if crs_auth is None:
                print("CRS authority code missing.")
                return None

            epsg_code = crs_auth[1]
            transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg_code}", always_xy=True)
            easting, northing = transformer.transform(lon, lat)

            # Filter for water points (classification 9)
            water_mask = las.classification == 9
            if not np.any(water_mask):
                print("No water-classified points found.")
                return None

            water_points = np.vstack((x[water_mask], y[water_mask])).T
            tree = cKDTree(water_points)
            dist, idx = tree.query([easting, northing])

            if dist > 100:
                print("No nearby water point found.")
                return None

            gpstimes = las.gps_time[water_mask]
            adjusted_time = gpstimes[idx]
            full_gps_time = adjusted_time + 1_000_000_000  # adjust to standard GPS time

            return full_gps_time

    except Exception as e:
        print(f"Error: {e}")
        return None


def est_dem_baseflow(stream_reach, source):
    """
        finds baseflow for a dem along a stream reach
    """
    # extract the lat and lon
    lat = stream_reach.latitude
    lon = stream_reach.longitude

    # get the date range of the lidar data
    # dem_dates = get_dem_dates(lat, lon)
    lidar_gpstime = find_water_gpstime(lat, lon)

    gpstime_date = None
    if lidar_gpstime:
        gpstime_date = gpstime_to_date(lidar_gpstime)

    # use the date range to estimate the baseflow
    if not gpstime_date:   # if no dates given, just use the median flow
        dem_baseflow = stream_reach.get_median_flow(source)
    else:               # if there are dates, use them lol
        dem_baseflow = stream_reach.get_flow_on_date(gpstime_date, source)
    print(f'the dem_baseflow is {dem_baseflow}')
    return dem_baseflow
