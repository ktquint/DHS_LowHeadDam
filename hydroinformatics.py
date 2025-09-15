import io
import os
import ast
import geoglows
import requests
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from datetime import date
from math import cos, radians
import matplotlib.pyplot as plt
from typing import List, Optional
from shapely.geometry import Point, box, LineString


# eventually i'll need a real api key i think... talk to sujan about that
nwm_api_key = "AIzaSyC4BXXMQ9KIodnLnThFi5Iv4y1fDR4U1II"
# import os
# nwm_api_key = (os.getenv('NWM_API_KEY'))
# if not nwm_api_key:
#     raise ValueError('NWM_API_KEY environment variable not set')


def make_bbox(latitude, longitude, distance_deg=0.5):
    """
        creates a bounding box around a point (lat, lon) ±distance_deg degrees.
    """
    lat_min = latitude - distance_deg
    lat_max = latitude + distance_deg
    lon_min = longitude - distance_deg / cos(radians(latitude))  # adjust for longitude convergence
    lon_max = longitude + distance_deg / cos(radians(latitude))
    return lon_min, lat_min, lon_max, lat_max


def haversine(lat1, lon1, lat2, lon2):
    """
        computes approximate distance (km) between two points.
    """
    from math import radians, sin, cos, sqrt, atan2
    R = 6371.0  # Earth radius in km
    d_lat = radians(lat2 - lat1)
    d_lon = radians(lon2 - lon1)
    a = sin(d_lat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(d_lon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


class StreamReach:
    """
        this class represents a stream reach.

        characteristics of a stream reach include:
            - geometry (the hydrography)
            - streamflow (discharge)
            - time (corresponding to streamflow)
            for our purposes we also want:
                - ID (which LHD does the stream reach correspond to?)
                - data sources (where does the geometry and streamflow come from?)

        to find the aforementioned information, we need:
            - lhd_id: the ID for this stream's LHD
            - latitude/longitude: where the LHD is located
            - data_sources: how do we want to estimate geometry and streamflow info
            - geoglows_streams: location of .gpkg containing GEOGLOWS hydrography
            - nwm_ds: zarr database that contains NWM info
            - streamflow (bool): You want to assign streamflow values to the stream reach
            - geometry (bool): You want to assign hydrography geometry to the stream reach
    """
    def __init__(self, lhd_id, latitude, longitude, data_sources, geoglows_streams=None,
                 nwm_ds=None, streamflow=True, geometry=True):
        self.id = lhd_id
        self.latitude = latitude
        self.longitude = longitude
        self.data_sources = data_sources
        # if using geoglows, we'll need a path to the stream gpkg
        self.geoglows_streams = geoglows_streams
        # if using nwm, we'll use a preloaded ds
        self.nwm_ds = nwm_ds
        # if true we'll get streamflow info
        self.streamflow = streamflow
        self.geometry = geometry

        self.flow_cms = None
        self.time = None
        self.name = None

        # GEOGLOWS Info
        self.geoglows_metadata = None
        self.geoglows_time = None
        self.geoglows_flow = None
        self.geoglows_geom = None
        self.linkno = None

        # NWM Info
        self.nwm_time = None
        self.nwm_flow = None
        self.nwm_geom = None
        self.reach_id = None

        # USGS Info
        self.usgs_time = None
        self.usgs_flow = None
        self.usgs_lat = None
        self.usgs_lon = None
        self.site_no = None
        self.usgs_geom =None

        self._load_data()


    def _load_data(self):
        valid_sources = False
        metadata_path = "./geoglows_metadata.parquet"

        if "GEOGLOWS" in self.data_sources:
            if not os.path.isfile(metadata_path):
                metadata_df = geoglows.data.metadata_table()
                metadata_df.to_parquet(metadata_path)
            self.geoglows_metadata = pd.read_parquet(metadata_path)
            self._load_geoglows()
            valid_sources = True

        if "National Water Model" in self.data_sources:
            self._load_nwm()
            valid_sources = True

        if "USGS" in self.data_sources:
            self._load_usgs()
            valid_sources = True

        if not valid_sources:
            raise ValueError("Unsupported data source. Use 'GEOGLOWS', 'National Water Model', or 'USGS'.")


    def _load_geoglows(self):
        # first we need to find the linkno based on the closest, highest order streamline
        # we'll use a bounding box so we don't have to load the whole thing
        bbox_coords = make_bbox(self.latitude, self.longitude, 0.002)
        bbox_geom = box(*bbox_coords)
        gdf = gpd.read_file(self.geoglows_streams, bbox=bbox_geom)
        print("finished reading in the geoglows streamlines")
        dam_point = Point(self.latitude, self.longitude)

        # calculate the distance to all streamlines in the bounding box
        gdf["distance"] = gdf.geometry.distance(dam_point)

        # of the closest streams, we only want the ones of the highest order
        max_strm_order = gdf['strmOrder'].max()
        highest_order_streams = gdf[gdf['strmOrder'] == max_strm_order]

        # now we'll find the closest stream
        nearest = highest_order_streams.loc[highest_order_streams['distance'].idxmin()]

        # save the linkno
        self.linkno = nearest["LINKNO"]

        if self.geometry:
            self.geoglows_geom = gpd.GeoDataFrame([nearest], crs=gdf.crs)

        if self.streamflow:
            df = geoglows.data.retrospective(river_id=self.linkno, bias_corrected=True)
            df.index = pd.to_datetime(df.index)

            df = df.reset_index().rename(columns={"index": "time", int(self.linkno): "flow_cms"})
            df = df.sort_values('time')

            # remove any negative numbers
            df = df[df['flow_cms'] >= 0]

            self.geoglows_flow = df['flow_cms']
            self.geoglows_time = df['time']


    def _load_nwm(self):
        r = requests.get(f"https://nwm-api.ciroh.org/geometry?lat={self.latitude}&lon={self.longitude}"
                         f"&output_format=csv&key={nwm_api_key}")

        # Check for successful response (HTTP status code 200)
        if r.status_code == 200:
            # Convert API response to pandas DataFrame
            df = pd.read_csv(io.StringIO(r.text))
            # Extract first (and only) reach ID from the response
            # print(df['station_id'].values)
            self.reach_id = df['station_id'].values[0]
        else:
            # Raise error if API request fails
            raise requests.exceptions.HTTPError(r.text)

        if self.geometry:
            url = f"https://nwm-api.ciroh.org/geometry?comids={self.reach_id}&key={nwm_api_key}"
            response = requests.get(url)

            if response.status_code != 200:
                raise ValueError(f"NWM API error {response.status_code}: {response.text}")

            data = response.json()[0]
            coords = data['geometry'].replace("LINESTRING(", "").replace(")", "")
            coord_pairs = [(float(x.split(" ")[0]), float(x.split(" ")[1])) for x in coords.split(", ")]
            linestring = LineString(coord_pairs)

            gdf = gpd.GeoDataFrame({'reach_id': [self.reach_id], 'source': ['NWM']}, geometry=[linestring], crs="EPSG:4326")
            self.nwm_geom = gdf

        if self.streamflow:
            # Select the feature_id and slice the time to the valid range
            stream = self.nwm_ds['streamflow'].sel(
                feature_id=int(self.reach_id),
                time=slice("1979-02-01", None))

            # Compute the selection and convert to a pandas DataFrame
            df = stream.compute().to_dataframe()

            # Rename the column and ensure the index is a datetime type
            df = df.rename(columns={"streamflow": "flow_cms"})
            df.index = pd.to_datetime(df.index)

            # Reset the index to make 'time' a column and assign attributes
            df = df.reset_index()
            # remove any negative numbers
            df = df[df['flow_cms'] >= 0]

            self.nwm_flow = df['flow_cms']
            self.nwm_time = df['time']


    def _load_usgs(self):
        # create bounding box coordinates
        bbox = make_bbox(self.latitude, self.longitude, 0.1)

        # Request sites with siteType=ST (surface water sites) that have instantaneous data
        bbox_url = (
            f"https://waterservices.usgs.gov/nwis/site/?format=rdb"
            f"&bBox={bbox[0]:.7f},{bbox[1]:.7f},{bbox[2]:.7f},{bbox[3]:.7f}"
            f"&siteType=ST&hasDataTypeCd=dv"
        )
        response = requests.get(bbox_url)
        data = response.text

        # Read tab-separated data, skipping header comments
        response_df = pd.read_csv(io.StringIO(data), sep="\t", comment="#", skip_blank_lines=True)
        response_df['dec_lat_va'] = pd.to_numeric(response_df['dec_lat_va'], errors='coerce')
        response_df['dec_long_va'] = pd.to_numeric(response_df['dec_long_va'], errors='coerce')
        response_df = response_df.dropna(subset=['dec_lat_va', 'dec_long_va'])

        # Filter for likely stream gages and calculate distance
        stream_df = response_df[response_df['site_no'].astype(str).str.len() <= 10].copy()

        stream_df['distance_km'] = stream_df.apply(
            lambda row: haversine(self.latitude, self.longitude, row['dec_lat_va'], row['dec_long_va']),
            axis=1)
        print(f'This is for LHD No. {self.id}')
        print("stream_df: ")
        print(stream_df[['site_no', 'station_nm', 'distance_km']].sort_values(by='distance_km'))

        # Prioritize sites with "river" in the name, with a fallback
        river_df = stream_df[stream_df['station_nm'].str.contains(r'river| R ', case=False, na=False)]

        if river_df.empty:
            nearest_site = stream_df.loc[stream_df['distance_km'].idxmin()]
        else:
            nearest_site = river_df.loc[river_df['distance_km'].idxmin()]

        print("nearest_site: ")
        print(nearest_site)
        self.name = nearest_site['station_nm']
        self.site_no = nearest_site['site_no']

        if self.geometry:
            # Store the lat/lon for the site
            self.usgs_lat = nearest_site['dec_lat_va']
            self.usgs_lon = nearest_site['dec_long_va']

            # Create a GeoDataFrame for the single point location of the USGS site
            self.usgs_geom = gpd.GeoDataFrame(
                [{'site_no': self.site_no, 'station_nm': self.name}],
                geometry=[Point(self.usgs_lon, self.usgs_lat)],
                crs="EPSG:4326"
            )

        if self.streamflow:
            start_date = '1850-01-01'
            end_date = date.today().isoformat()

            # Fetch daily values for discharge (00060)
            url = (
                f"https://waterservices.usgs.gov/nwis/dv/?sites={self.site_no}"
                f"&parameterCd=00060&startDT={start_date}&endDT={end_date}&format=json"
            )
            response = requests.get(url)
            data = response.json()

            # Check for empty time series response
            if not data['value']['timeSeries']:
                print(f"No USGS streamflow data found for site {self.site_no}.")
                return

            try:
                df = pd.json_normalize(data['value']['timeSeries'][0]['values'][0]['value'])
                df = df.rename(columns={'dateTime': 'time', 'value': 'flow_cfs'})
                df['flow_cfs'] = pd.to_numeric(df['flow_cfs'], errors='coerce')

                # remove any negative numbers
                df = df[df['flow_cfs'] >= 0]

                self.usgs_flow = df['flow_cfs'] / 35.315
                self.usgs_time = pd.to_datetime(df['time'])
            except (KeyError, IndexError):
                print(f"Could not parse streamflow JSON for site {self.site_no}.")


    def get_median_flow(self, source):
        if source == "GEOGLOWS":
            return float(self.geoglows_flow.median())
        elif source == "USGS":
            return float(self.usgs_flow.median())
        else:
            return float(self.nwm_flow.median())


    def get_median_flow_in_range(self, start_date, end_date, source):
        """
            returns the median streamflow (m³/s) for the given date range as a float.
        """
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        if source == "GEOGLOWS":
            df = pd.DataFrame({
                'time': self.geoglows_time,
                'flow_cms': self.geoglows_flow,
            })

        elif source == "USGS":
            df = pd.DataFrame({
                'time': self.usgs_time,
                'flow_cms': self.usgs_flow,
            })

        else:
            df = pd.DataFrame({
                'time': self.nwm_time,
                'flow_cms': self.nwm_flow,
            })

        df = df.set_index('time')
        filtered_df = df.loc[start_date:end_date]

        if filtered_df.empty:
            return np.nan
        else:
            median_flow = filtered_df['flow_cms'].median()
            return float(median_flow)


    def get_flow_on_date(self, target_date, source):
        """
            returns the streamflow (m³/s) for the given date as a float.
            if no flow is associated with the date, it returns None.
        """
        time_data, flow_data = None, None
        if source == "GEOGLOWS":
            time_data = self.geoglows_time
            flow_data = self.geoglows_flow

        elif source == "USGS":
            time_data = self.usgs_time
            flow_data = self.usgs_flow

        elif source == "National Water Model":
            time_data = self.nwm_time
            flow_data = self.nwm_flow

        if time_data is None or flow_data is None:
            return None

        df = pd.DataFrame({
            'time': time_data,
            'flow_cms': flow_data,
        })

        target_date = pd.to_datetime(target_date)
        match = df[df['time'].dt.date == target_date.date()]

        if not match.empty:
            return float(match.iloc[0]['flow_cms'])


    def plot_hydrographs(self):
        # let's go through each source and plot a hydrograph
        for source in self.data_sources:
            plt.figure(figsize=(12, 6))
            if source == "GEOGLOWS":
                comid = self.linkno
                time = self.geoglows_time
                flow = self.geoglows_flow
            elif source == "USGS":
                comid = self.site_no
                time = self.usgs_time
                flow = self.usgs_flow
            else: # source == NWM
                comid = self.reach_id
                time = self.nwm_time
                flow = self.nwm_flow

            if time is not None and flow is not None:
                plt.plot(time, flow, linewidth=1)
                plt.title(f"Streamflow Hydrograph - {source} ID {comid}")
                plt.xlabel("Date")
                plt.ylabel("Streamflow (m³/s)")
                plt.grid(True)
                plt.tight_layout()
                plt.show()

    def plot_fdcs(self):
        # let's go through each source and plot a hydrograph
        plt.figure(figsize=(10, 6))
        for source in self.data_sources:
            if source == "GEOGLOWS":
                comid = self.linkno
                flow_data = self.geoglows_flow
            elif source == "USGS":
                comid = self.site_no
                flow_data = self.usgs_flow
            else: # source == NWM
                comid = self.reach_id
                flow_data = self.nwm_flow

            if flow_data is not None:
                flow_data = flow_data.dropna()
                exceedance, sorted_flows = _calculate_fdc(flow_data)

                plt.plot(exceedance, sorted_flows)
                plt.xscale('linear')
                plt.yscale('log')
                plt.title(f"Flow Duration Curve - {source} ID {comid}")
                plt.xlabel("Exceedance Probability (%)")
                plt.ylabel("Streamflow (m³/s)")
                plt.grid(True, which="both", linestyle='--')
                plt.tight_layout()
                plt.show()


    def export_fdcs(self):
        fdc_results = {}

        for source in self.data_sources:
            if source == "GEOGLOWS":
                flow_data = self.geoglows_flow
            elif source == "USGS":
                flow_data = self.usgs_flow
            else: # source == NWM
                flow_data = self.nwm_flow

            if flow_data is not None:
                flow_data = flow_data.dropna()
                exceedance, sorted_flows = _calculate_fdc(flow_data)
                fdc_results[source] = (exceedance, sorted_flows)

        return fdc_results

def compare_hydrographs(reach):
    """
        plots hydrographs from multiple data sources for a given reach.
    """
    plt.figure(figsize=(12, 6))

    # Use a dictionary to map data sources to their attributes
    source_map = {
        "GEOGLOWS": (reach.geoglows_time, reach.geoglows_flow, reach.linkno),
        "USGS": (reach.usgs_time, reach.usgs_flow, reach.site_no),
        "National Water Model": (reach.nwm_time, reach.nwm_flow, reach.reach_id),
    }

    for source in reach.data_sources:
        if source in source_map:
            time, flow, data_id = source_map[source]

            if time is not None and flow is not None:
                plt.plot(time, flow, label=f"{source} ({data_id})")

    # Set title based on whether USGS data was plotted
    title_name = reach.name if "USGS" in reach.data_sources else "Hydrograph Comparison"
    plt.title(f"{title_name} (LHD No. {reach.id})")

    plt.xlabel("Date")
    plt.ylabel("Streamflow (m³/s)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def _calculate_fdc(flow_data):
    """
        helper function to calculate FDC.
    """
    flow_data = flow_data.dropna()
    sorted_flows = np.sort(flow_data)[::-1]
    ranks = np.arange(1, len(sorted_flows) + 1)
    exceedance = 100 * ranks / (len(sorted_flows) + 1)
    return exceedance, sorted_flows



def compare_fdcs(reach):
    """
        plots FDCs from multiple data sources for a given reach.
    """
    plt.figure(figsize=(10, 6))

    # Use a dictionary to map data sources to their attributes
    source_map = {
        "GEOGLOWS": (reach.geoglows_flow, reach.linkno),
        "USGS": (reach.usgs_flow, reach.site_no),
        "National Water Model": (reach.nwm_flow, reach.reach_id),
    }

    for source in reach.data_sources:
        if source in source_map:
            flow, data_id = source_map[source]

            if flow is not None:
                # Use the helper function for FDC calculation
                exceedance, sorted_flows = _calculate_fdc(flow)
                plt.plot(exceedance, sorted_flows, label=f"{source} ({data_id})")

    # Set title
    title_name = reach.name if "USGS" in reach.data_sources else "Flow Duration Curve"
    plt.title(f"{title_name} Comparison (LHD No. {reach.id})")

    plt.yscale('log')
    plt.xlabel("Exceedance Probability (%)")
    plt.ylabel("Streamflow (m³/s)")
    plt.grid(True, which="both", linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.show()


def create_multilayer_gpkg(
        gdfs: List[gpd.GeoDataFrame],
        output_path: str,
        layer_names: Optional[List[str]] = None
) -> None:
    """
        Saves a list of GeoDataFrames to a single GeoPackage file, with each
        GeoDataFrame as its own layer.

        Args:
            gdfs (List[gpd.GeoDataFrame]): A list of GeoDataFrames to save.
            output_path (str): The file path for the output GeoPackage.
                               Must end with '.gpkg'.
            layer_names (Optional[List[str]], optional): A list of names for each
                                                         layer. If not provided,
                                                         layers will be named
                                                         'layer_1', 'layer_2', etc.
                                                         Defaults to None.

        Raises:
            ValueError: If the output path is not a .gpkg file.
            ValueError: If the number of layer names does not match the number
                        of GeoDataFrames.
            ValueError: If the list of GeoDataFrames is empty.
    """
    # --- 1. Input Validation ---
    if not output_path.lower().endswith('.gpkg'):
        raise ValueError("Output file path must end with '.gpkg'")

    if not gdfs:
        raise ValueError("The list of GeoDataFrames cannot be empty.")

    if layer_names:
        if len(gdfs) != len(layer_names):
            raise ValueError(
                "The number of layer names must match the number of GeoDataFrames."
            )
    else:
        # Create default layer names if none are provided
        layer_names = [f"layer_{i + 1}" for i in range(len(gdfs))]

    # --- 2. Remove Existing File
    # This ensures you start with a fresh file on each run.
    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"Removed existing file at: {output_path}")

    # --- 3. Write Each GeoDataFrame to a Layer ---
    print(f"Creating GeoPackage at: {output_path}")
    for gdf, layer_name in zip(gdfs, layer_names):
        if not isinstance(gdf, gpd.GeoDataFrame) or gdf.empty:
            print(f"Warning: Skipping empty or invalid GeoDataFrame for layer '{layer_name}'.")
            continue

        try:
            gdf.to_file(output_path, driver="GPKG", layer=layer_name)
            print(f"  - Successfully wrote layer: '{layer_name}'")
        except Exception as e:
            print(f"  - Failed to write layer '{layer_name}'. Error: {e}")

    print("GeoPackage creation complete.")


def create_gpkg_from_lists(
    lists_of_gdfs: List[List[gpd.GeoDataFrame]],
    output_path: str,
    layer_names: Optional[List[str]] = None
) -> None:
    """
        Combines lists of GeoDataFrames and saves each combined list as a
        separate layer in a single GeoPackage file.

        Args:
            lists_of_gdfs (List[List[gpd.GeoDataFrame]]): A list where each
                element is another list of GeoDataFrames to be merged.
            output_path (str): The file path for the output GeoPackage.
            layer_names (Optional[List[str]], optional): A list of names for each
                output layer. Defaults to None.
    """
    merged_gdfs = []
    print("--- Merging lists of GeoDataFrames ---")
    for i, gdf_list in enumerate(lists_of_gdfs):
        if not gdf_list:
            print(f"Warning: Inner list at index {i} is empty, skipping.")
            continue

        # Ensure all elements are GeoDataFrames before concatenating
        if not all(isinstance(g, gpd.GeoDataFrame) for g in gdf_list):
            print(f"Warning: Inner list at index {i} contains non-GeoDataFrame elements, skipping.")
            continue

        # We assume all gdfs in a list share the same CRS and take it from the first one.
        crs = gdf_list[0].crs
        # Use pandas.concat to merge the list of GeoDataFrames
        merged_gdf = gpd.GeoDataFrame(
            pd.concat(gdf_list, ignore_index=True), crs=crs
        )
        merged_gdfs.append(merged_gdf)
        print(f"Merged list {i+1} into a single GeoDataFrame with {len(merged_gdf)} features.")

    # Now call the original function with the list of merged GeoDataFrames
    create_multilayer_gpkg(merged_gdfs, output_path, layer_names)
