import io
import os
import s3fs
import zipfile
import requests
import pandas as pd
import geopandas as gpd
from shapely.geometry import box, Point
from download_dem import sanitize_filename
from math import radians, sin, cos, sqrt, atan2


def make_bbox(latitude: float, longitude: float, distance_deg: float=0.5) -> list[float]:
    """
        creates a bounding box around a point (lat, lon) ±distance_deg degrees.
    """
    lat_min = latitude - distance_deg
    lat_max = latitude + distance_deg
    lon_min = longitude - distance_deg / cos(radians(latitude))  # adjust for longitude convergence
    lon_max = longitude + distance_deg / cos(radians(latitude))
    return [lon_min, lat_min, lon_max, lat_max]


def haversine(lat1, lon1, lat2, lon2):
    """
        computes approximate distance (km) between two points.
    """
    R = 6371.0  # Earth radius in km
    d_lat = radians(lat2 - lat1)
    d_lon = radians(lon2 - lon1)
    a = sin(d_lat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(d_lon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def find_huc(latitude: float, longitude: float) -> str:
    # we'll use a bounding box to find a streamgage near our dam
    bbox = make_bbox(latitude, longitude, 0.1)
    # Request sites with siteType=ST (surface water sites)
    bbox_url = (f"https://waterservices.usgs.gov/nwis/site/?format=rdb"
                f"&bBox={bbox[0]:.7f},{bbox[1]:.7f},{bbox[2]:.7f},{bbox[3]:.7f}"
                f"&siteType=ST")

    response = requests.get(bbox_url)
    data = response.text

    # Read data into DataFrame
    response_df = pd.read_csv(io.StringIO(data), sep="\t", comment="#", skip_blank_lines=True)

    # Convert lat/lon columns to numeric
    response_df['dec_lat_va'] = pd.to_numeric(response_df['dec_lat_va'], errors='coerce')
    response_df['dec_long_va'] = pd.to_numeric(response_df['dec_long_va'], errors='coerce')

    # Drop rows with missing coordinates
    response_df = response_df.dropna(subset=['dec_lat_va', 'dec_long_va'])

    # Filter to short site numbers (likely surface water gages)
    stream_df = response_df[response_df['site_no'].astype(str).str.len() <= 8].copy()

    # Now find the closest among these
    stream_df['distance_km'] = stream_df.apply(lambda row:haversine(latitude, longitude,
                                                                    row['dec_lat_va'], row['dec_long_va']),
                                               axis=1)

    nearest_site = stream_df.loc[stream_df['distance_km'].idxmin()]

    return nearest_site['huc_cd'][:4]


def download_NHDPlus(latitude: float, longitude: float, flowline_dir: str) -> str|None:
    """
        finds the HUC4 associated with the stream
        downloads the NHDPlus HR data for that HUC
        merges the Flowline with VAA table
        saves new Flowline layer as a .gpkg and deletes the old one
    """
    hu4 = find_huc(latitude, longitude)

    # before we go any further, let's make sure we haven't already downloaded this huc...
    already_downloaded = [f for f in os.listdir(flowline_dir) if hu4 in f]
    if already_downloaded:
        print(f"The NHD Flowline for HU4: {hu4} is already downloaded")
        # return the file name to store in the DataFrame
        return os.path.join(flowline_dir, already_downloaded[0])

    bbox = make_bbox(latitude, longitude, 0.01)

    product = "National Hydrography Dataset Plus High Resolution (NHDPlus HR)"
    base_url = "https://tnmaccess.nationalmap.gov/api/v1/products"

    params = {"bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
              "datasets": product, "max": 10,
              "format": "GeoPackage, NHDPlus HR Rasters",
              "outputFormat": "JSON",}

    try:
        # ask the api to grant our wish
        response = requests.get(base_url, params=params)
        response.raise_for_status()

        # let's look at the results
        results = response.json().get("items", [])
        print(f'original results: {results}')

        # filter for .gpkg files
        gpkg_results = [item for item in results if
                        'gpkg' in item.get("downloadURL", "").lower() or
                        'geopackage'  in item.get("title", "").lower()]

        print(f'gpkg results: {gpkg_results}')

        # filter for the right huc
        huc_results = [item for item in gpkg_results if
                       f'_{hu4}_' in item.get("downloadURL", "").lower()]


        if len(huc_results) > 1:
            print("Retrieved too many results...")
            return None

        final_gpkg = huc_results[0]

        os.makedirs(flowline_dir, exist_ok=True)

        title = final_gpkg.get("title", "Unnamed")
        sanitized_title = sanitize_filename(title)
        download_url = final_gpkg.get("downloadURL", "")

        gpkg_loc = os.path.join(flowline_dir, download_url.rsplit('/', 1)[-1].replace('.zip', '.gpkg'))

        if not os.path.exists(gpkg_loc):
            local_zip_path = os.path.join(flowline_dir, f"{sanitized_title}.zip")
            print(f"Retrieving {sanitized_title}...")

            with requests.get(download_url, stream=True) as r:
                r.raise_for_status()
                with open(local_zip_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            print(f"Saved to {local_zip_path}")

            # unzip the zip file
            with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
                zip_ref.extractall(flowline_dir)

            # let's remove the zip file after extraction
            os.remove(local_zip_path)


        # assuming layers are always named 'NHDFlowline' and 'NHDPlusFlowlineVAA' instead of lowercase
        flowlines_gdf = gpd.read_file(filename=gpkg_loc, layer='NHDFlowline', engine='fiona')
        metadata_gdf = gpd.read_file(filename=gpkg_loc, layer='NHDPlusFlowlineVAA', engine='fiona')

        # normalize column names to lowercase
        flowlines_gdf.columns = flowlines_gdf.columns.str.lower()
        metadata_gdf.columns = metadata_gdf.columns.str.lower()

        # merge gdf on uniform fields
        merged_gdf = flowlines_gdf.merge(metadata_gdf, on=['nhdplusid', 'reachcode', 'vpuid'])

        # save the gdf to a new file
        new_gpkg_loc = gpkg_loc.replace('.gpkg', '_VAA.gpkg')
        merged_gdf.to_file(filename=new_gpkg_loc, layer='NHDFlowline', driver='GPKG')

        # delete the old files
        xml_loc = gpkg_loc.replace('.gpkg', '.xml')
        jpg_loc = gpkg_loc.replace('.gpkg', '.jpg')
        old_locs = [gpkg_loc, xml_loc, jpg_loc]
        for loc in old_locs:
            os.remove(loc)

        return new_gpkg_loc

    except requests.RequestException as e:
        print(e)


def download_TDXHYDRO(latitude: float, longitude: float, flowline_dir: str, TDX_full: str) -> str:
    """
        Downloads the GeoGLows GPKGs locally.
    """
    # first we need to find the linkno based on the closest streamline
    # we'll use a bounding box so we don't have to load the whole thing
    base_url = 'geoglows-v2/hydrography/'

    bbox_coords = make_bbox(latitude, longitude, 0.003)
    bbox_geom = box(*bbox_coords)
    gdf = gpd.read_file(TDX_full, bbox=bbox_geom)
    print("finished reading in the geoglows streamlines")
    dam_point = Point(latitude, longitude)

    # find the nearest streamline to our dam
    gdf["distance"] = gdf.geometry.distance(dam_point)
    nearest = gdf.loc[gdf["distance"].idxmin()]

    # save the comid
    vpu_code = nearest["VPUCode"]

    gpkg_loc = os.path.join(flowline_dir, f"streams_{vpu_code}.gpkg")
    gpkg_name = gpkg_loc.split("/")[-1]
    vpu_id = gpkg_name[-8:-5]
    gpkg_url = f"{base_url}vpu={vpu_id}/{gpkg_name}"

    if not os.path.exists(gpkg_loc):
        try:
            fs = s3fs.S3FileSystem(anon=True)
            # Download a file
            with fs.open(gpkg_url, 'rb') as f_in:
                with open(gpkg_loc, 'wb') as f_out:
                    f_out.write(f_in.read())

            return gpkg_loc

        except requests.exceptions.RequestException as e:
            print(f"Failed to download {gpkg_url}: {e}")
    else:
        print(f"Local file {gpkg_loc} already exists")
        return gpkg_loc
