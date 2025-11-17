import os
import ast
import pandas as pd
from core import hydroinformatics as hi
from .download_dem import download_dem
from .dem_baseflow import est_dem_baseflow
from .download_flowline import download_NHDPlus, download_TDXHYDRO


class Dam:
    """
        Dam object created from a row of a .csv file
    """
    def __init__(self, **kwargs):
        """
            kwargs will be a .csv row turned into a dictionary
        """
        # database info
        self.ID = int(kwargs['ID'])
        self.name = kwargs.get('name', None)

        # geographical info
        self.latitude = float(kwargs['latitude'])
        self.longitude = float(kwargs['longitude'])
        self.city = kwargs.get('city', None)
        self.county = kwargs.get('county', None)
        self.state = kwargs.get('state', None)

        # fatality info
        self.fatality_dates = ast.literal_eval(kwargs['fatality_dates'])

        # physical information
        self.weir_length = float(kwargs['weir_length'])

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


    def assign_flowlines(self, flowline_dir: str, VPU_gpkg: str):
        # download the flowlines based on the provided source
        print(f"Assigning flowlines based on {self.hydrography}")

        if self.hydrography == 'NHDPlus':
            self.flowline_NHD = download_NHDPlus(self.latitude, self.longitude, flowline_dir)

        elif self.hydrography == 'GEOGLOWS':
            self.flowline_TDX = download_TDXHYDRO(self.latitude, self.longitude, flowline_dir, VPU_gpkg)

    def assign_dem(self, dem_dir, resolution):
        # Call download_dem, which now returns resolution_meters
        dem_subdir, self.final_titles, resolution_meters = download_dem(
            self.ID, self.latitude, self.longitude, self.weir_length, dem_dir, resolution
        )

        # Clear existing paths first to avoid incorrect assignments if resolution changes
        self.dem_1m = None
        self.dem_3m = None
        self.dem_10m = None
        self.final_resolution = None  # Reset this too

        if dem_subdir and resolution_meters is not None:
            # Assign path based on the returned resolution_meters
            # Add some tolerance for floating point comparisons
            if resolution_meters <= 1.5:
                self.dem_1m = dem_subdir
                self.final_resolution = "Digital Elevation Model (DEM) 1 meter"  # Set descriptive name
            elif resolution_meters <= 5.0:  # e.g., 1/9 arc-second ~ 3m
                self.dem_3m = dem_subdir
                self.final_resolution = "National Elevation Dataset (NED) 1/9 arc-second"
            else:  # Assume >= 10m for 1/3 arc-second or others
                self.dem_10m = dem_subdir
                self.final_resolution = "National Elevation Dataset (NED) 1/3 arc-second Current"
            print(
                f"Assigned DEM path '{dem_subdir}' to appropriate resolution category based on ~{resolution_meters:.2f}m")
        elif dem_subdir:
            # Handle case where resolution couldn't be determined but path exists
            print(
                f"Warning: DEM path '{dem_subdir}' exists, but resolution could not be determined. Assigning based on preferred input '{resolution}'.")
            # Fallback to assigning based on the initially requested resolution string
            if "1 meter" in resolution:
                self.dem_1m = dem_subdir
            elif "1/9 arc-second" in resolution:
                self.dem_3m = dem_subdir
            else:
                self.dem_10m = dem_subdir
        else:
            print(f"DEM assignment failed for Dam {self.ID}.")

    def create_reach(self, nwm_ds=None, tdx_vpu_map=None):
        """
        Instantiates a StreamReach object based on the dam's
        streamflow (hydrology) and flowline (hydrography) settings.
        """
        print(f'Creating Stream Reach for Dam No. {self.ID}')

        # 1. Build the data_sources list from *both* attributes
        # (These are set by assign_hydrology and assign_hydrography)
        data_sources = [self.hydrology]  # self.hydrology is the streamflow source
        if self.hydrography not in data_sources:
            data_sources.append(self.hydrography)  # self.hydrography is the flowline source

        # 2. Check if *either* source requires the GEOGLOWS map
        geoglows_map_path = None
        if 'GEOGLOWS' in data_sources:
            if tdx_vpu_map:
                geoglows_map_path = tdx_vpu_map
            else:
                # Fallback to the old attribute if it exists, though tdx_vpu_map is preferred
                geoglows_map_path = getattr(self, 'flowline_TDX', None)

            if geoglows_map_path is None:
                print(f"Warning: GEOGLOWS source selected for Dam {self.ID} but VPU map path not found.")

        # 3. Create the StreamReach object with correct flags
        self.dam_reach = hi.StreamReach(
            lhd_id=self.ID,
            latitude=self.latitude,
            longitude=self.longitude,
            data_sources=data_sources,  # Pass both sources
            geoglows_streams=geoglows_map_path,  # FIX 2: Pass map if *either* source is GEOGLOWS
            nwm_ds=nwm_ds,
            streamflow=True  # <-- We always want to get streamflow
        )


    def set_dem_baseflow(self):
        """
            estimates the DEM baseflow if it hasn't been calculated yet...
        """
        print("Estimating DEM baseflow...")
        # 1. Determine the attribute name to check based on hydrology
        baseflow_attr = None
        if self.hydrology == 'National Water Model':
            baseflow_attr = 'dem_baseflow_NWM'
        elif self.hydrology == 'GEOGLOWS':
            baseflow_attr = 'dem_baseflow_GEOGLOWS'

        # 2. Now, run the logic a single time using the dynamic attribute name
        if baseflow_attr:
            # Get the current value of the attribute (e.g., self.dem_baseflow_NWM)
            current_value = getattr(self, baseflow_attr)

            if pd.isna(current_value):
                print(f"{baseflow_attr} is not set. Calling estimation function for Dam ID: {self.ID}")
                baseflow = est_dem_baseflow(self.dam_reach, self.hydrology)
                # Set the attribute (e.g., self.dem_baseflow_NWM = baseflow)
                setattr(self, baseflow_attr, baseflow)
            else:
                print(f"{baseflow_attr} already has a value: {current_value}")
        else:
            # This else handles cases where hydrology is not NWM or GEOGLOWS
            print(f"Skipping baseflow check: hydrology '{self.hydrology}' is not recognized.")


    def set_fatal_flows(self):
        print("Estimating fatal flows...")
        fatality_dates_kept = []  # Renamed to avoid confusion
        fatality_flows_kept = []  # Renamed
        skipped_dates_reasons = {}  # Dictionary to store skipped dates and reasons

        for fatality_date in self.fatality_dates:  # Loop through original dates
            fatality_flow_result = self.dam_reach.get_flow_on_date(fatality_date, self.hydrology)

            if isinstance(fatality_flow_result, float):  # Check if it's a valid number
                fatality_dates_kept.append(fatality_date)
                fatality_flows_kept.append(fatality_flow_result)
            else:
                # Store the reason why the date was skipped
                skipped_dates_reasons[fatality_date] = fatality_flow_result
                print(
                    f"Skipping date {fatality_date} for dam {self.ID}. Reason: {fatality_flow_result}")  # Optional: print why it was skipped

        # Optional: Print a summary of skipped dates
        if skipped_dates_reasons:
            print(f"\nSummary of skipped dates for Dam {self.ID}:")
            for date, reason in skipped_dates_reasons.items():
                print(f"  - {date}: {reason}")
            print("-" * 20)

        # Update attributes with the lists of dates/flows that were successfully retrieved
        self.fatality_dates = fatality_dates_kept  # Overwrite with only the dates that had flow data
        if self.hydrology == 'National Water Model':
            # Only update if the original was NaN, otherwise keep existing data
            if pd.isna(self.fatality_flows_NWM) or not self.fatality_flows_NWM:  # Check if None or empty list
                self.fatality_flows_NWM = fatality_flows_kept
        elif self.hydrology == 'GEOGLOWS':
            # Only update if the original was NaN, otherwise keep existing data
            if pd.isna(self.fatality_flows_GEOGLOWS) or not self.fatality_flows_GEOGLOWS:  # Check if None or empty list
                self.fatality_flows_GEOGLOWS = fatality_flows_kept


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
