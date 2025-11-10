import io
import os
import geoglows
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
from math import cos, radians
import matplotlib.pyplot as plt
from typing import List, Optional
from shapely.geometry import Point, box, LineString


# env_path = Path(__file__).parent.parent / 'config' / '.env'
# load_dotenv(dotenv_path=env_path)
#
# nwm_api_key = os.getenv("API_KEY")

nwm_api_key = "AIzaSyC4BXXMQ9KIodnLnThFi5Iv4y1fDR4U1II"

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
        self.geoglows_streams_path = geoglows_streams
        self.nwm_ds = nwm_ds
        self.fetch_streamflow = streamflow
        self.fetch_geometry = geometry

        # Private attributes for caching results
        self._geoglows_df = None
        self._nwm_df = None
        self._reach_id = None
        self._linkno = None
        self._site_no = None
        self._geoglows_geom = None
        self._nwm_geom = None

    @property
    def linkno(self):
        """Lazy-loads the GEOGLOWS link number."""
        if self._linkno is None:
            self._load_geoglows_reach()
        return self._linkno

    @property
    def reach_id(self):
        """Lazy-loads the NWM reach ID."""
        if self._reach_id is None:
            self._load_nwm_reach()
        return self._reach_id

    @property
    def geoglows_df(self):
        """Lazy-loads and caches the GEOGLOWS streamflow DataFrame."""
        if self._geoglows_df is None and "GEOGLOWS" in self.data_sources and self.fetch_streamflow:
            self._geoglows_df = self._load_geoglows_flow()
        return self._geoglows_df

    @property
    def nwm_df(self):
        """Lazy-loads and caches the NWM streamflow DataFrame."""
        if self._nwm_df is None and "National Water Model" in self.data_sources and self.fetch_streamflow:
            self._nwm_df = self._load_nwm_flow()
        return self._nwm_df

    @property
    def geoglows_geom(self):
        """Lazy-loads and caches the GEOGLOWS geometry."""
        if self._geoglows_geom is None and "GEOGLOWS" in self.data_sources and self.fetch_geometry:
            # Ensure linkno is loaded first
            _ = self._linkno
            self._load_geoglows_reach(force_reload=False)  # _load_geoglows_reach populates the geometry
        return self._geoglows_geom

    @property
    def nwm_geom(self):
        """Lazy-loads and caches the NWM geometry."""
        if self._nwm_geom is None and "National Water Model" in self.data_sources and self.fetch_geometry:
            _ = self.reach_id  # Ensure reach_id is loaded
            self._load_nwm_reach(force_reload=False)  # _load_nwm_reach populates the geometry
        return self._nwm_geom

    def _load_geoglows_reach(self, force_reload=True):
        """Finds the nearest GEOGLOWS reach and caches its ID and geometry."""
        if self._linkno and not force_reload:
            return

        bbox_coords = make_bbox(self.latitude, self.longitude, 0.002)
        bbox_geom = box(*bbox_coords)
        gdf = gpd.read_file(self.geoglows_streams_path, bbox=bbox_geom)
        dam_point = Point(self.latitude, self.longitude)

        gdf["distance"] = gdf.geometry.distance(dam_point)
        max_strm_order = gdf['strmOrder'].max()
        highest_order_streams = gdf[gdf['strmOrder'] == max_strm_order]
        nearest = highest_order_streams.loc[highest_order_streams['distance'].idxmin()]

        self._linkno = nearest["LINKNO"]
        if self.fetch_geometry:
            self._geoglows_geom = gpd.GeoDataFrame([nearest], crs=gdf.crs)

    def _load_nwm_reach(self, force_reload=True):
        """Finds the nearest NWM reach and caches its ID and geometry."""
        if self._reach_id and not force_reload:
            return

        # Fetch reach_id
        r = requests.get(f"https://nwm-api.ciroh.org/geometry?lat={self.latitude}&lon={self.longitude}"
                         f"&output_format=csv&key={nwm_api_key}")
        if r.status_code != 200:
            raise requests.exceptions.HTTPError(r.text)
        df = pd.read_csv(io.StringIO(r.text))
        self._reach_id = df['station_id'].values[0]

        # Fetch geometry if requested
        if self.fetch_geometry:
            url = f"https://nwm-api.ciroh.org/geometry?comids={self._reach_id}&key={nwm_api_key}"
            response = requests.get(url)
            if response.status_code != 200:
                raise ValueError(f"NWM API error {response.status_code}: {response.text}")

            data = response.json()[0]
            coords = data['geometry'].replace("LINESTRING(", "").replace(")", "")
            coord_pairs = [(float(x.split(" ")[0]), float(x.split(" ")[1])) for x in coords.split(", ")]
            linestring = LineString(coord_pairs)
            self._nwm_geom = gpd.GeoDataFrame({'reach_id': [self._reach_id]}, geometry=[linestring], crs="EPSG:4326")

    def _load_geoglows_flow(self):
        """Fetches and processes GEOGLOWS streamflow data."""
        df = geoglows.data.retrospective(river_id=self._linkno, bias_corrected=True)
        df.index = pd.to_datetime(df.index)
        df = df.reset_index().rename(columns={"index": "time", int(self._linkno): "flow_cms"})
        df = df.sort_values('time')
        df = df[df['flow_cms'] >= 0]
        return df.set_index('time')

    def _load_nwm_flow(self):
        """Fetches and processes NWM streamflow data."""
        if self.nwm_ds is None:
            raise ValueError("NWM dataset (nwm_ds) must be provided for NWM streamflow.")
        stream = self.nwm_ds['streamflow'].sel(feature_id=int(self.reach_id), time=slice("1979-02-01", None))
        df = stream.compute().to_dataframe().rename(columns={"streamflow": "flow_cms"})
        df.index = pd.to_datetime(df.index)
        df = df[df['flow_cms'] >= 0]
        return df

    def get_flow_on_date(self, target_date, source):
        """More efficient method to get flow on a specific date, returning reasons for failure."""
        df = self._get_source_df(source)
        if df is None or df.empty:
            print(f"Dam {self.id}: No data available for source: {source}")
            return np.nan

        # Convert target_date to datetime.date for comparison
        try:
            target_dt = pd.to_datetime(target_date).date()
        except ValueError:
            print(f"Dam {self.id}: Invalid target date format for {target_date}")
            return np.nan

        # Check if the date is within the range of the DataFrame index
        min_date = df.index.min().date()
        max_date = df.index.max().date()

        if not (min_date <= target_dt <= max_date):
            print(f"Dam {self.id}: Date {target_dt} out of range ({min_date} to {max_date})")
            return np.nan

        # Look for the specific date
        match = df[df.index.date == target_dt]

        if not match.empty:
            # Check if the flow value itself is valid (e.g., not NaN)
            flow_value = match.iloc[0]['flow_cms']
            if pd.notna(flow_value):
                return float(flow_value)  # Success! Return the flow
            else:
                print(f"Dam {self.id}: Data exists for {target_dt}, but flow value is missing (NaN)")
                return np.nan
        else:
            # This case might be less common if the date range check passes,
            # but could happen if there are gaps in daily data within the range.
            print(f"Dam {self.id}: No data found for specific date {target_dt} within the range")
            return np.nan

    def get_median_flow(self, source: str) -> float:
        """
        Calculates the median flow for the entire period of record for a given source.

        Args:
            source (str): The data source ('GEOGLOWS', 'National Water Model', or 'USGS').

        Returns:
            float: The median streamflow in m^3/s, or np.nan if data is unavailable.
        """
        df = self._get_source_df(source)
        if df is not None and not df.empty:
            return float(df['flow_cms'].median())
        return np.nan

    def get_median_flow_in_range(self, start_date, end_date, source):
        """More efficient method to get median flow in a date range."""
        df = self._get_source_df(source)
        if df is None: return np.nan

        filtered_df = df.loc[start_date:end_date]
        return float(filtered_df['flow_cms'].median()) if not filtered_df.empty else np.nan

    def _get_source_df(self, source):
        """Helper to get the correct DataFrame based on the source string."""
        if source == "GEOGLOWS":
            return self.geoglows_df
        elif source == "National Water Model":
            return self.nwm_df
        else:
            raise ValueError(f"Unknown source: {source}")

    ### RESTORED AND UPDATED METHODS ###

    def plot_hydrographs(self):
        """Plots a hydrograph for each available data source."""
        for source in self.data_sources:
            df = self._get_source_df(source)
            if df is not None and not df.empty:
                if source == "GEOGLOWS":
                    comid = self._linkno
                elif source == "USGS":
                    comid = self._site_no
                else:
                    comid = self.reach_id

                plt.figure(figsize=(12, 6))
                plt.plot(df.index, df['flow_cms'], linewidth=1)
                plt.title(f"Streamflow Hydrograph - {source} ID {comid}")
                plt.xlabel("Date")
                plt.ylabel("Streamflow (m³/s)")
                plt.grid(True)
                plt.tight_layout()
                plt.show()

    def plot_fdcs(self):
        """Plots a Flow Duration Curve for each available data source on a single plot."""
        plt.figure(figsize=(10, 6))
        for source in self.data_sources:
            df = self._get_source_df(source)
            if df is not None and not df.empty:
                if source == "GEOGLOWS":
                    comid = self._linkno
                elif source == "USGS":
                    comid = self._site_no
                else:
                    comid = self.reach_id

                flow_data = df['flow_cms'].dropna()
                exceedance, sorted_flows = _calculate_fdc(flow_data)

                plt.plot(exceedance, sorted_flows, label=f"{source} ({comid})")

        plt.yscale('log')
        plt.title(f"Flow Duration Curve Comparison (LHD No. {self.id})")
        plt.xlabel("Exceedance Probability (%)")
        plt.ylabel("Streamflow (m³/s)")
        plt.grid(True, which="both", linestyle='--')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def export_fdcs(self):
        """Exports FDC data for each source to a dictionary."""
        fdc_results = {}
        for source in self.data_sources:
            df = self._get_source_df(source)
            if df is not None and not df.empty:
                flow_data = df['flow_cms'].dropna()
                exceedance, sorted_flows = _calculate_fdc(flow_data)
                fdc_results[source] = (exceedance.tolist(), sorted_flows.tolist())
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
