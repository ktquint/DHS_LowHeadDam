import xarray as xr
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
        self.name = kwargs['name']

        # geographical info
        self.latitude = kwargs['latitude']
        self.longitude = kwargs['longitude']
        self.city = kwargs['city']
        self.county = kwargs['county']
        self.state = kwargs['state']

        # fatality info
        self.fatality_dates = kwargs['fatality_dates']

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



    def assign_flowlines(self, flowline_dir: str, hydrography_source: str, TDX_full: str=None):
        # download the flowlines based on the provided source
        print(f"Assigning flowlines based on {hydrography_source}")

        if hydrography_source == 'NHDPlus':
            self.flowline_NHD = download_NHDPlus(self.latitude, self.longitude, flowline_dir)

        elif hydrography_source == 'GEOGLOWS':
            self.flowline_TDX = download_TDXHYDRO(self.latitude, self.longitude, flowline_dir, TDX_full)


    def assign_dem(self, dem_dir, resolution):
        dem_subdir, self.final_titles, self.final_resolution = download_dem(self.ID, self.latitude, self.longitude, self.weir_length, dem_dir, resolution)

        if self.final_resolution == "Digital Elevation Model (DEM) 1 meter":
            self.dem_1m = dem_subdir
        elif self.final_resolution == "National Elevation Dataset (NED) 1/9 arc-second":
            self.dem_3m = dem_subdir
        else:
            self.dem_10m = dem_subdir


    def est_dem_baseflow(self, hydrology):
        print("Estimating DEM baseflow...")

        geoglows_streams = None
        nwm_ds = None
        if hydrology == 'National Water Model':
            s3_path = 's3://noaa-nwm-retrospective-2-1-zarr-pds/chrtout.zarr'
            nwm_ds = xr.open_zarr(s3_path, consolidated=True, storage_options={"anon": True})
        elif hydrology == 'GEOGLOWS':
            geoglows_streams = self.flowline_TDX
        else:
            print("This should never happen")
            return None

        dam_reach = hi.StreamReach(self.ID, self.latitude, self.longitude, [hydrology], geoglows_streams,
                                   nwm_ds, streamflow=True, geometry=False)
        if hydrology == 'National Water Model':
            self.dem_baseflow_NWM = est_dem_baseflow(dam_reach, hydrology)
        elif hydrology == 'GEOGLOWS':
            self.dem_baseflow_GEOGLOWS = est_dem_baseflow(dam_reach, hydrology)



    def assign_output(self, output_dir: str):
        self.output_dir = output_dir


    def __repr__(self):
        return "Hi, I'm a Dam"



