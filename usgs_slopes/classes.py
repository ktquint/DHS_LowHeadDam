import os
import io
import requests
import zipfile
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box
from stream_slope import make_bbox, sanitize_filename


class StreamGage:
    def __init__(self, station_id: str):
        # given info
        self.id = station_id

        # get name and other info...
        self.name = None
        self.lat = None
        self.lon = None
        self.huc = None
        self.geom = None

        self._get_nwis_info()

        # gdfs for merging metadata
        self.gpkg_loc = None
        self.hydrography = None
        self.metadata = None
        self.slope = None

        # other potentially useful info...
        self.nhdplusid = None


    def _get_nwis_info(self):
        url = f"https://waterservices.usgs.gov/nwis/site/?format=rdb&sites={self.id}&siteOutput=basic"
        response = requests.get(url)
        data = response.text

        # Skip the 1st line (field widths) + comment lines
        df = pd.read_csv(io.StringIO(data), sep="\t", comment="#", skiprows=2, skip_blank_lines=True)
        df = df.iloc[1:].reset_index(drop=True)

        df = df[df['agency_cd'] != 'agency_cd']  # drop repeat headers if present

        df['dec_lat_va'] = pd.to_numeric(df['dec_lat_va'], errors='coerce')
        df['dec_long_va'] = pd.to_numeric(df['dec_long_va'], errors='coerce')


        self.lat = df['dec_lat_va'].iloc[0]
        self.lon = df['dec_long_va'].iloc[0]
        self.name = df['station_nm'].iloc[0]
        self.huc = df['huc_cd'].values[0][:4] # just get first 4 characters

        self.geom = gpd.GeoDataFrame(data=[{'site_no': self.id, 'station_nm': self.name}],
                                          geometry=[Point(self.lon, self.lat)], crs="EPSG:4326")


    def download_nhd(self, download_dir):
        """find the geo-database that contains the hydrography around a streamgage"""
        bbox = make_bbox(self.lat, self.lon, 0.0003)

        product = "National Hydrography Dataset Plus High Resolution (NHDPlus HR)"
        base_url = "https://tnmaccess.nationalmap.gov/api/v1/products"

        params = {"bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
                  "datasets": product,
                  "max": 10,  # number of results to return
                  "prodFormat": "GeoPackage, NHDPlus HR Rasters",
                  "outputFormat": "JSON"}

        try:
            # query the API
            response = requests.get(base_url, params=params)
            response.raise_for_status()

            # parse the results
            results = response.json().get("items", [])

            # filter to just .gpkg files
            gpkg_results = [item for item in results if
                            'gpkg' in item.get("downloadURL", "").lower() or
                            'geopackage' in item.get("title", "").lower()]

            # filter to just the huc of your streamgage
            huc_results = [item for item in gpkg_results if
                            f'_{self.huc}_' in item.get("downloadURL", "").lower()]

            # there should just be one results
            if len(huc_results) > 1:
                print(f"Retrieved too many results...")
                return

            final_gpkg = huc_results[0]

            os.makedirs(download_dir, exist_ok=True)

            title = final_gpkg.get("title", "Unnamed")
            sanitized_title = sanitize_filename(title)  # sanitize the file name
            download_url = final_gpkg.get("downloadURL")

            self.gpkg_loc = os.path.join(download_dir, download_url.rsplit('/', 1)[-1].replace('.zip', '.gpkg'))

            if not os.path.exists(self.gpkg_loc):
                local_zip_path = os.path.join(download_dir, f"{sanitized_title}.zip")
                print(f"Retrieving {sanitized_title}...")

                with requests.get(download_url, stream=True) as r:
                    r.raise_for_status()
                    with open(local_zip_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                print(f"Saved to {local_zip_path}")

                # unzip the zip file
                with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
                    zip_ref.extractall(download_dir)

                # let's remove the zip file after extraction
                os.remove(local_zip_path)
            else:
                print(f"GPKG already downloaded at {self.gpkg_loc}")


        except requests.RequestException as e:
            print(e)

    def merge_metadata(self):
        # now we'll read in the hydrography layer and save it to our object
        bbox = make_bbox(self.lat, self.lon, 0.01)
        bbox_geom = box(*bbox)

        all_flowlines = gpd.read_file(filename=self.gpkg_loc, layer='NHDFlowline', bbox=bbox_geom,
                                      engine='fiona')
        all_flowlines.columns = all_flowlines.columns.str.lower()  # normalize column names to lowercase
        all_flowlines = all_flowlines.to_crs(epsg=4326)

        # Reproject both geometries to a projected CRS
        projected_crs = "EPSG:26918"  # NAD83 / UTM zone 18N
        all_flowlines_proj = all_flowlines.to_crs(projected_crs)
        gage_geom_proj = self.geom.to_crs(projected_crs).geometry.iloc[0]

        # Calculate distance in meters
        all_flowlines_proj["distance"] = all_flowlines_proj.geometry.distance(gage_geom_proj)

        # Find the nearest line
        nearest_flowline = all_flowlines_proj.loc[[all_flowlines_proj["distance"].idxmin()]]

        self.hydrography = nearest_flowline

        self.nhdplusid = int(nearest_flowline['nhdplusid'].values[0])

        # Read metadata layer and normalize column names
        metadata = gpd.read_file(filename=self.gpkg_loc, layer='NHDPlusFlowlineVAA', engine='fiona')
        metadata.columns = metadata.columns.str.lower()

        # Only keep relevant columns (if present)
        required_cols = ['nhdplusid', 'slope']
        metadata = metadata[[col for col in required_cols if col in metadata.columns]]

        self.metadata = metadata

        joined_data = self.hydrography.merge(self.metadata, on='nhdplusid')

        self.slope = joined_data['slope'].values[0] if 'slope' in joined_data.columns else None


    def __repr__(self):
        return (f"Station ID: {self.id}\n"
                f"Station Name: {self.name}\n"
                f"Latitude: {self.lat}\n"
                f"Longitude: {self.lon}\n"
                f"4-digit HUC: {self.huc}\n"
                f"Slope: {self.slope}")



james = StreamGage(station_id="10163000")

james.download_nhd("C:/Users/ki87ujmn/Downloads")
james.merge_metadata()

print(james)
