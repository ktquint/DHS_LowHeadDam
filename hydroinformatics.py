import io
import ast
import geoglows
import requests
import numpy as np
import pandas as pd
import xarray as xr
import seaborn as sns
import geopandas as gpd
from math import cos, radians
import matplotlib.pyplot as plt
from datetime import datetime, date
from shapely.geometry import Point, box


nwm_api_key = "AIzaSyC4BXXMQ9KIodnLnThFi5Iv4y1fDR4U1II"
# import os
# nwm_api_key = (os.getenv('NWM_API_KEY'))
# if not nwm_api_key:
#     raise ValueError('NWM_API_KEY environment variable not set')


def make_bbox(latitude, longitude, distance_deg=0.5):
    """Creates a bounding box around a point (lat, lon) ±distance_deg degrees."""
    lat_min = latitude - distance_deg
    lat_max = latitude + distance_deg
    lon_min = longitude - distance_deg / cos(radians(latitude))  # adjust for longitude convergence
    lon_max = longitude + distance_deg / cos(radians(latitude))
    return lon_min, lat_min, lon_max, lat_max


def haversine(lat1, lon1, lat2, lon2):
    """Computes approximate distance between two points (km)."""
    from math import radians, sin, cos, sqrt, atan2
    R = 6371.0  # Earth radius in km
    d_lat = radians(lat2 - lat1)
    d_lon = radians(lon2 - lon1)
    a = sin(d_lat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(d_lon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


class StreamReach:
    def __init__(self, lhd_id, latitude, longitude, data_source, geoglows_streams=None, nwm_ds=None, streamflow=True):
        self.id = lhd_id
        self.latitude = latitude
        self.longitude = longitude
        self.data_source = data_source
        # if using geoglows, we'll need a path to the stream gpkg
        self.geoglows_streams = geoglows_streams
        # if using nwm, we'll use a preloaded ds
        self.nwm_ds = nwm_ds
        # if true we'll get streamflow info
        self.streamflow = streamflow

        self.df = None
        self.flow_cms = None
        self.time = None
        self.name = None

        # all the different comid's
        self.reach_id = None
        self.site_no = None
        self.linkno = None
        # the comid we'll print later
        self.comid = None

        self._load_data()

    @staticmethod
    def _format_dates(date_list):
        formatted = []
        for d in date_list:
            try:
                dt = datetime.strptime(d.strip(), "%Y-%m-%d")
                formatted.append(dt.strftime("%Y-%m-%dT%H:%M:%S"))
            except ValueError:
                continue
        return formatted

    def _load_data(self):
        if self.data_source == "GEOGLOWS":
            self._load_geoglows()
            self.comid = self.linkno
        elif self.data_source == "National Water Model":
            self._load_nwm()
            self.comid = self.reach_id
        elif self.data_source == "USGS":
            self._load_usgs()
            self.comid = self.site_no
        else:
            raise ValueError("Unsupported data source. Use 'GEOGLOWS', 'National Water Model', or 'USGS'.")

        if self.df is not None and not self.df.empty:
            self.df = self.df.sort_values('time')
            self.flow_cms = self.df['flow_cms']
            self.time = self.df['time']

    def _load_geoglows(self):
        # first we need to find the linkno based on the closest streamline
        # we'll use a bounding box so we don't have to load the whole thing
        # ... it seems like this gets less accurate the larger the bounding box...
        bbox_coords = make_bbox(self.latitude, self.longitude, 0.002)
        bbox_geom = box(*bbox_coords)
        gdf = gpd.read_file(self.geoglows_streams, bbox=bbox_geom)
        print("finished reading in the geoglows streamlines")
        dam_point = Point(self.latitude, self.longitude)

        # find the nearest streamline to our dam
        gdf["distance"] = gdf.geometry.distance(dam_point)
        # print(gdf[['LINKNO', 'distance']].head())
        nearest = gdf.loc[gdf["distance"].idxmin()]

        # save the comid
        self.linkno = nearest["LINKNO"]

        comid = int(self.linkno)
        df = geoglows.data.retrospective(river_id=comid, bias_corrected=True)
        df.index = pd.to_datetime(df.index)
        if self.streamflow:
            self.df = df.reset_index().rename(columns={"index": "time", comid: "flow_cms"})

    def _load_nwm(self):
        r = requests.get(f"https://nwm-api.ciroh.org/geometry?lat={self.latitude}&lon={self.longitude}"
                         f"&output_format=csv&key={nwm_api_key}")
        # Check for successful response (HTTP status code 200)
        if r.status_code == 200:
            # Convert API response to pandas DataFrame
            df = pd.read_csv(io.StringIO(r.text))
            # Extract first (and only) reach ID from the response
            # print(df['station_id'].values)
            reach_id = df['station_id'].values[0]
        else:
            # Raise error if API request fails
            raise requests.exceptions.HTTPError(r.text)

        self.reach_id = reach_id
        if self.streamflow:
            stream = self.nwm_ds['streamflow'].sel(feature_id=int(self.reach_id))
            valid_start = "1979-02-01T00:00:00"
            stream = stream.sel(time=slice(valid_start, None))

            computed_stream = stream.compute()  # this loads the data into memory before converting to data_frame

            df = computed_stream.to_dataframe().reset_index()
            df = df.rename(columns={"streamflow": "flow_cms"})
            self.df = df[['time', 'flow_cms']].copy()
            self.df['time'] = pd.to_datetime(self.df['time'])

    def _load_usgs(self):
        # create bounding box coordinates
        bbox = make_bbox(self.latitude, self.longitude, 0.3)

        # Request sites with siteType=ST (surface water sites)
        bbox_url = (
            f"https://waterservices.usgs.gov/nwis/site/?format=rdb"
            f"&bBox={bbox[0]:.7f},{bbox[1]:.7f},{bbox[2]:.7f},{bbox[3]:.7f}"
            f"&siteType=ST"
        )
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
        stream_df = response_df[response_df['site_no'].astype(str).str.len() <= 10].copy()

        # Now find the closest among these
        stream_df['distance_km'] = stream_df.apply(
            lambda row: haversine(self.latitude, self.longitude, row['dec_lat_va'], row['dec_long_va']),
            axis=1
        )
        #
        # sorted_streams = stream_df.sort_values(by='distance_km', ascending=True).head()
        #
        # print(sorted_streams[['station_nm', 'site_no', 'distance_km']])

        river_df = stream_df[stream_df['station_nm'].str.contains(r'river| R ', case=False, na=False)]
        # print(f"This info is for LHD No. {self.id}")
        # print(river_df['station_nm'].head())
        # print(stream_df.sort_values(by='distance_km', ascending=True).head())
        if river_df.empty:
            nearest_site = stream_df.loc[stream_df['distance_km'].idxmin()]
        else:
            nearest_site = river_df.loc[river_df['distance_km'].idxmin()]
        print(nearest_site[['station_nm', 'site_no', 'distance_km']])
        self.name = nearest_site['station_nm']

        # print("Site Latitude:", nearest_site['dec_lat_va'])
        # print("Site Longitude:", nearest_site['dec_long_va'])

        self.site_no = nearest_site['site_no']

        if self.streamflow:
            start_date = '1850-01-01'
            end_date = date.today().isoformat()

            url = (
                f"https://waterservices.usgs.gov/nwis/dv/?sites={self.site_no}"
                f"&parameterCd=00060&startDT={start_date}&endDT={end_date}&format=json"
            )
            response = requests.get(url)
            data = response.json()

            if not data['value']['timeSeries']:
                print(f"No USGS streamflow data found for site {self.site_no}.")
                self.df = pd.DataFrame()
                return

            records = []
            for ts in data['value']['timeSeries']:
                for value in ts['values'][0]['value']:
                    records.append({
                        'time': value['dateTime'][:10],
                        'flow_cfs': float(value['value']) if value['value'] != '' else None
                    })

            df = pd.DataFrame(records)
            df['flow_cms'] = df['flow_cfs'].apply(lambda x: x / 35.315 if pd.notnull(x) else None)
            df['time'] = pd.to_datetime(df['time'])
            self.df = df[['time', 'flow_cms']]

    def get_median_flow(self):
        if self.flow_cms is not None and not self.flow_cms.empty:
            return float(self.flow_cms.median())
        return None

    def get_median_flow_in_range(self, start_date, end_date):
        """
        Returns the median streamflow (m³/s) for the given date range as a float.
        """
        if self.df is None or self.df.empty:
            return None

        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        mask = (self.df['time'] >= start_date) & (self.df['time'] <= end_date)
        flows_in_range = self.df.loc[mask]['flow_cms']

        if not flows_in_range.empty:
            return float(flows_in_range.median())
        return None

    def get_flows_in_range(self, start_date, end_date=None):
        """
        Returns a DataFrame of streamflow (m³/s) for the given date or date range.
        If not found, returns an empty DataFrame.
        """
        if self.df is None or self.df.empty:
            return pd.DataFrame()

        start_date = pd.to_datetime(start_date)
        if end_date:
            end_date = pd.to_datetime(end_date)
            mask = (self.df['time'] >= start_date) & (self.df['time'] <= end_date)
        else:
            mask = self.df['time'].dt.date == start_date.date()

        return self.df.loc[mask]


    def get_flow_on_date(self, target_date):
        """
        Returns the streamflow (m³/s) for the given date as a float.
        If not found, returns None.
        """
        if self.df is None or self.df.empty:
            return None

        target_date = pd.to_datetime(target_date)
        match = self.df[self.df['time'].dt.date == target_date.date()]

        if not match.empty:
            return float(match.iloc[0]['flow_cms'])
        return None

    def plot_hydrograph(self):
        if self.df is None or self.df.empty:
            print("No streamflow data available.")
            return

        plt.figure(figsize=(12, 6))
        plt.plot(self.time, self.flow_cms, linewidth=1)
        plt.title(f"Streamflow Hydrograph - {self.data_source} ID {self.comid}")
        plt.xlabel("Date")
        plt.ylabel("Streamflow (m³/s)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_fdc(self):
        if self.flow_cms is None or self.flow_cms.empty:
            print("No flow data to plot FDC.")
            return

        flow_data = self.flow_cms.dropna()
        sorted_flows = np.sort(flow_data)[::-1]
        ranks = np.arange(1, len(sorted_flows) + 1)
        exceedance = 100 * ranks / (len(sorted_flows) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(exceedance, sorted_flows)
        plt.xscale('linear')
        plt.yscale('log')
        plt.title(f"Flow Duration Curve - {self.data_source} ID {self.comid}")
        plt.xlabel("Exceedance Probability (%)")
        plt.ylabel("Streamflow (m³/s)")
        plt.grid(True, which="both", linestyle='--')
        plt.tight_layout()
        plt.show()


def compare_hydrographs(reaches):
    plt.figure(figsize=(12, 6))

    data_sources = []
    name = None
    lhd_id = None

    for reach in reaches:
        data_sources.append(reach.data_source)
        plt.plot(reach.time, reach.flow_cms, label=f"{reach.data_source} ({reach.comid})")

        if reach.data_source == "USGS":
            name = reach.name
            lhd_id = reach.id

    if "USGS" in data_sources:
        plt.title(f"Hydrograph Comparison for {name} (LHD No. {lhd_id})")
    else:
        plt.title(f"Hydrograph Comparison")

    plt.xlabel("Date")
    plt.ylabel("Streamflow (m³/s)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def compare_fdcs(reaches):
    plt.figure(figsize=(10, 6))

    data_sources = []
    name = None
    lhd_id = None

    for reach in reaches:
        data_sources.append(reach.data_source)
        flow_data = reach.flow_cms.dropna()
        sorted_flows = np.sort(flow_data)[::-1]
        ranks = np.arange(1, len(sorted_flows) + 1)
        exceedance = 100 * ranks / (len(sorted_flows) + 1)

        plt.plot(exceedance, sorted_flows, label=f"{reach.data_source} ({reach.comid})")
        if reach.data_source == "USGS":
            name = reach.name
            lhd_id = reach.id

    if "USGS" in data_sources:
        plt.title(f"FDC Comparison for {name} (LHD No. {lhd_id})")
    else:
        plt.title(f"Flow Duration Curve Comparison")


    plt.xscale('linear')
    plt.yscale('log')
    plt.xlabel("Exceedance Probability (%)")
    plt.ylabel("Streamflow (m³/s)")
    plt.grid(True, which="both", linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_discharge_comparisons(df):
    """
    Creates a faceted bar chart to compare discharge estimates.
    """
    # This creates a grid of plots.
    # Each row is a different dam, and each column is a different date.
    g = sns.catplot(
        data=df,
        kind='bar',
        x='method',  # Estimation method on the x-axis
        y='discharge_cms',  # Discharge value on the y-axis
        col='date',  # A new column of plots for each unique date
        row='dam_id',  # A new row of plots for each unique dam
        height=4,
        aspect=1.2,
        sharey=False  # Allow y-axis to be different for each plot
    )

    # Improve readability
    g.set_axis_labels("Estimation Method", "Discharge (m³/s)")
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    g.fig.suptitle('Discharge Estimates on Fatality Dates', y=1.03)
    plt.tight_layout()
    plt.show()


def main(lhd_csv_path, streams_gpkg_path, ):
    lhd_df = pd.read_csv(lhd_csv_path)
    print("finished reading in the LHD Database...")

    # load the nwm ds once
    # sources = ["GEOGLOWS", "National Water Model", "USGS"]
    sources = ["GEOGLOWS", "USGS"]

    ds = None
    if "National Water Model" in sources:
        s3_path = 's3://noaa-nwm-retrospective-2-1-zarr-pds/chrtout.zarr'
        ds = xr.open_zarr(s3_path, consolidated=True, storage_options={"anon": True})
        print("finished reading the Zarr Dataset...")

    results_list = []

    for index, row_i in lhd_df.iterrows():
        lat = row_i['latitude']
        lon = row_i['longitude']
        dam_id = row_i['ID']

        # let's make stream reach objects for each source

        stream_reaches = []

        for source in sources:
            try:
                # Create the object as you did before
                reach = StreamReach(dam_id, lat, lon, source,
                                    geoglows_streams=streams_gpkg_path, nwm_ds=ds, streamflow=False)

                # Only append the object to the list if it has a non-empty DataFrame
                stream_reaches.append(reach)
                results_list.append({'dam_id': dam_id,
                                     'method': source,
                                     'comid': reach.comid})
                print(f"created {source} StreamReach object")
                # if reach.df is not None and not reach.df.empty:
                #     stream_reaches.append(reach)
                #     results_list.append({'dam_id': dam_id,
                #                          'method': source,
                #                          'comid': reach.comid})
                #     print(f"created {source} StreamReach object")
                # else:
                #     print(f"INFO: Skipping {source} for point ({lat}, {lon}) due to no data.")

            except Exception as e:
                print(f"ERROR: Failed to process {source} for point ({lat}, {lon}). Reason: {e}")

        # Only call the comparison functions if you have data to compare
        # if len(stream_reaches) > 1:
        #     compare_hydrographs(stream_reaches)
        #     compare_fdcs(stream_reaches)
        # elif len(stream_reaches) == 1:
        #     stream = stream_reaches[0]
        #     stream.plot_hydrograph()
        #     stream.plot_fdc()

        # now let's look at each estimate for flow on the fatality dates
        # fatality_dates = ast.literal_eval(row_i['fatality_dates'])
        #
        # for date_i in fatality_dates:
        #     # let's look at the flows on these days...
        #     for stream in stream_reaches:
        #         flow = stream.get_flow_on_date(date_i)
        #         if flow is not None:
        #             results_list.append({'dam_id': dam_id,
        #                                  'date': pd.to_datetime(date_i),
        #                                  'method': stream.data_source,
        #                                  'comid': stream.comid,
        #                                  'discharge_cms': flow})
        #
        #
        #             print(f"{stream.data_source} Streamflow on {date_i}:")
        #             print(stream.get_flow_on_date(date_i))

    results_df = pd.DataFrame(results_list)
    results_df.to_csv("E:/LowHead_Dam_Streamflow.csv", index=False)



if __name__ == "__main__":
    main("E:/LowHead_Dam_Database.csv", "E:/TDX_HYDRO/streams.gpkg")


# import geoglows
# import numpy as np
# import pandas as pd
# import xarray as xr
# from dateutil import parser     # date/time parsing from strings
# from datetime import datetime, timedelta    # provides timedelta objects for representing time differences or durations
#
#
# # format dates into 'YYYY-MM-DDThh:mm:ss' for NWM
# def format_dates(date_list):
#     formatted = []
#     for date in date_list:
#         try:
#             dt = datetime.strptime(date.strip(), "%Y-%m-%d")
#             formatted.append(dt.strftime("%Y-%m-%dT%H:%M:%S"))
#         except ValueError:
#             continue  # skip bad formats
#     return formatted
#
# def get_streamflow(data_source, comid, date_range=None):
#     """
#         comid is either a GEOGLOWS LINKNO or National Water Model ReachID
#
#         date_range the [start_date, end_date] for which flow data will be retrieved.
#             it's okay for start_date == end_date
#         needs to be in the format "%Y-%m-%d"
#
#         returns average streamflow—for the entire record if no lat-long is given, else it's the average from the dates the lidar was taken
#     """
#     try:
#         comid = int(comid)
#     except ValueError:
#         raise ValueError("comid needs to be an int")
#
#     start_date, end_date = None, None
#
#     if date_range:
#         start_date = date_range[0]
#         end_date = date_range[1]
#
#     print(start_date, end_date)
#
#     # this is all the data for the comid
#     if data_source == "GEOGLOWS":
#         # this is all the data for the comid
#         historic_df = geoglows.data.retrospective(river_id=comid, bias_corrected=True)
#         historic_df.index = pd.to_datetime(historic_df.index)
#
#         if not date_range:
#             return np.median(historic_df[comid])
#
#         else:
#             subset_df = historic_df.loc[date_range[0]:date_range[1]]
#             return np.median(subset_df[comid])
#
#     elif data_source == "National Water Model":
#         s3_path = 's3://noaa-nwm-retrospective-2-1-zarr-pds/chrtout.zarr'
#         nwm_ds = xr.open_zarr(s3_path, consolidated=True, storage_options={"anon": True})
#         variable_name = 'streamflow'
#         valid_start = "1979-02-01T00:00:00" # first available data
#
#         if not date_range:
#             reach_ds = nwm_ds[variable_name].sel(feature_id=comid)
#
#         else:
#             if start_date == end_date:
#                 start_date = format_dates([start_date])[0]
#                 if start_date < valid_start:
#                     return None
#                 end_date = (parser.parse(start_date) + timedelta(hours=23)).isoformat()
#
#             else:
#                 converted_dates = format_dates(date_range)
#                 start_date = converted_dates[0]
#                 end_date = converted_dates[-1]
#
#             reach_ds = nwm_ds[variable_name].sel(feature_id=comid).loc[dict(time=slice(start_date, end_date))]
#
#         df = reach_ds.to_dataframe().reset_index()
#         df = df.set_index('time')
#         df.index = pd.to_datetime(df.index)
#
#         streamflow = df['streamflow']
#         if not streamflow.empty:
#             return float(streamflow.mean())
#
#
# # # tested with same dates, different dates, and no dates... they all work
# # same_dates = ['2010-04-12', '2010-04-12']
# # dif_dates = ['2010-04-12', '2010-05-12']
# # linkno = 760645077
# # geo = "GEOGLOWS"
# # print(get_streamflow(geo, linkno, same_dates))
# # print(get_streamflow(geo, linkno, dif_dates))
# # print(get_streamflow(geo, linkno))
# #
# # reach_id = 368123
# # nwm = "National Water Model"
# # print(get_streamflow(nwm, reach_id, same_dates))
# # print(get_streamflow(nwm, reach_id, dif_dates))
# # print(get_streamflow(nwm, reach_id))



"""
as of now it still downloads the one in ohio wrong... 
it does when bounding box = 0.001, but doesn't catch a different stream at that level
"""