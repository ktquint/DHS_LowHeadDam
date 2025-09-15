import os
import ast
import pandas as pd
import hydroinformatics as hi
from download_dem import download_dem
from dem_baseflow import est_dem_baseflow
from download_flowline import download_NHDPlus, download_TDXHYDRO


class Dam:
    """
        Dam object created from a row of a .xlsx workbook
    """
    def __init__(self, **kwargs):
        """
            kwargs will be a .xlsx row turned into a dictionary
        """
        # database info
        self.ID = kwargs['ID']
        self.name = kwargs.get('name', None)

        # geographical info
        self.latitude = kwargs['latitude']
        self.longitude = kwargs['longitude']
        self.city = kwargs.get('city', None)
        self.county = kwargs.get('county', None)
        self.state = kwargs.get('state', None)

        # fatality info
        self.fatality_dates = ast.literal_eval(kwargs['fatality_dates'])

        # physical information
        self.weir_length = kwargs['weir_length']

        # optional info that you may already have
        # i'm making so many different fields because I want to be able to store as much info as possible
        # without overwriting anything
        self.dem_1m = kwargs.get('dem_1m', None)
        self.dem_3m = kwargs.get('dem_3m', None)
        self.dem_10m = kwargs.get('dem_10m', None)

        self.output_dir = kwargs.get('output_dir', None)

        self.flowline_NHD = kwargs.get('flowline_NHD', None)
        self.flowline_TDX = kwargs.get('flowline_TDX', None)

        self.final_titles = kwargs.get('final_titles', None)
        self.final_resolution = kwargs.get('final_resolution', None)

        self.dem_baseflow_NWM = kwargs.get('dem_baseflow_NWM', None)
        self.dem_baseflow_GEOGLOWS = kwargs.get('dem_baseflow_GEOGLOWS', None)

        # reset these attributes, so we can change them later
        self.hydrology = None
        self.hydrography = None
        self.dam_reach = None
        self.fatality_flows_NWM = kwargs.get('fatality_flows_NWM', None)
        self.fatality_flows_GEOGLOWS = kwargs.get('fatality_flows_GEOGLOWS', None)


    def assign_flowlines(self, flowline_dir: str, TDX_full: str="E:/TDX_HYDRO/streams.gpkg"):
        # download the flowlines based on the provided source
        print(f"Assigning flowlines based on {self.hydrography}")

        if self.hydrography == 'NHDPlus':
            self.flowline_NHD = download_NHDPlus(self.latitude, self.longitude, flowline_dir)

        elif self.hydrography == 'GEOGLOWS':
            self.flowline_TDX = download_TDXHYDRO(self.latitude, self.longitude, flowline_dir, TDX_full)


    def assign_dem(self, dem_dir, resolution):
        dem_subdir, self.final_titles, self.final_resolution = download_dem(self.ID, self.latitude, self.longitude, self.weir_length, dem_dir, resolution)

        if self.final_resolution == "Digital Elevation Model (DEM) 1 meter":
            self.dem_1m = dem_subdir
        elif self.final_resolution == "National Elevation Dataset (NED) 1/9 arc-second":
            self.dem_3m = dem_subdir
        else:
            self.dem_10m = dem_subdir


    def create_reach(self, nwm_ds=None):
        print(f'Creating Stream Reach for Dam No. {self.ID}')
        geoglows_streams = None
        if self.hydrology == 'GEOGLOWS':
            geoglows_streams = self.flowline_TDX

        self.dam_reach = hi.StreamReach(self.ID, self.latitude, self.longitude, [self.hydrology], geoglows_streams,
                                   nwm_ds, streamflow=True, geometry=False)


    def est_dem_baseflow(self):
        print("Estimating DEM baseflow...")
        # the reason why I still have to pass hydrology to hi.est_dem_baseflow is because the stream reach object could
        # have several hydrology options saved to it

        if self.hydrology == 'National Water Model' and self.dem_baseflow_NWM is None:
            self.dem_baseflow_NWM = est_dem_baseflow(self.dam_reach, self.hydrology)
        elif self.hydrology == 'GEOGLOWS' and self.dem_baseflow_GEOGLOWS is None:
            self.dem_baseflow_GEOGLOWS = est_dem_baseflow(self.dam_reach, self.hydrology)


    def est_fatal_flows(self):
        print("Estimating fatal flows...")
        fatality_dates = []
        fatality_flows = []

        for fatality_date in self.fatality_dates:
            fatality_flow = self.dam_reach.get_flow_on_date(fatality_date, self.hydrology)
            if fatality_flow is not None:
                fatality_dates.append(fatality_date)
                fatality_flows.append(fatality_flow)

        # we're remaking fatality dates because some of the dates may fall out of the range available with NWM
        self.fatality_dates = fatality_dates
        if self.hydrology == 'National Water Model' and self.fatality_flows_NWM is None:
            self.fatality_flows_NWM = fatality_flows
        elif self.hydrology == 'GEOGLOWS' and self.fatality_flows_GEOGLOWS is None:
            self.fatality_flows_NWM = fatality_flows


    def assign_output(self, output_dir: str):
        self.output_dir = output_dir


    def assign_hydrology(self, hydrology: str):
        self.hydrology = hydrology


    def assign_hydrography(self, hydrography: str):
        self.hydrography = hydrography


    def assign_reach(self, stream_reach):
        self.dam_reach = stream_reach


    def fdc_to_csv(self) -> None:
        fdc_results = self.dam_reach.export_fdcs()
        fdc_df = pd.DataFrame()

        for source, (exceedance, flows) in fdc_results.items():
            if source == self.hydrology:
                fdc_df = pd.DataFrame(
                    {
                        "Exceedance (%)": exceedance,
                        "Flow (cms)": flows
                    })
        flow_dir = os.path.join(self.output_dir, str(self.ID), "FLOW")
        os.makedirs(flow_dir, exist_ok=True)
        csv_path = os.path.join(flow_dir, f"{self.ID}_{self.hydrology}_FDC.csv")
        fdc_df.to_csv(str(csv_path), index=False)


    def __repr__(self):
        return "Hi, I'm a Dam"
