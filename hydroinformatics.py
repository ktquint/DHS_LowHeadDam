import geoglows
import numpy as np
import pandas as pd
import xarray as xr
from dateutil import parser     # date/time parsing from strings
from datetime import datetime, timedelta    # provides timedelta objects for representing time differences or durations


# format dates into 'YYYY-MM-DDThh:mm:ss' for NWM
def format_dates(date_list):
    formatted = []
    for date in date_list:
        try:
            dt = datetime.strptime(date.strip(), "%Y-%m-%d")
            formatted.append(dt.strftime("%Y-%m-%dT%H:%M:%S"))
        except ValueError:
            continue  # skip bad formats
    return formatted

def get_streamflow(data_source, comid, date_range=None):
    """
        comid is either a GEOGLOWS LINKNO or National Water Model ReachID

        date_range the [start_date, end_date] for which flow data will be retrieved.
            it's okay for start_date == end_date
        needs to be in the format "%Y-%m-%d"

        returns average streamflow—for the entire record if no lat-long is given, else it's the average from the dates the lidar was taken
    """
    try:
        comid = int(comid)
    except ValueError:
        raise ValueError("comid needs to be an int")

    start_date, end_date = None, None

    if date_range:
        start_date = date_range[0]
        end_date = date_range[1]

    print(start_date, end_date)

    # this is all the data for the comid
    if data_source == "GEOGLOWS":
        # this is all the data for the comid
        historic_df = geoglows.data.retrospective(river_id=comid, bias_corrected=True)
        historic_df.index = pd.to_datetime(historic_df.index)

        if not date_range:
            return np.median(historic_df[comid])

        else:
            subset_df = historic_df.loc[date_range[0]:date_range[1]]
            return np.median(subset_df[comid])

    elif data_source == "National Water Model":
        s3_path = 's3://noaa-nwm-retrospective-2-1-zarr-pds/chrtout.zarr'
        nwm_ds = xr.open_zarr(s3_path, consolidated=True, storage_options={"anon": True})
        variable_name = 'streamflow'
        valid_start = "1979-02-01T00:00:00" # first available data

        if not date_range:
            reach_ds = nwm_ds[variable_name].sel(feature_id=comid)

        else:
            if start_date == end_date:
                start_date = format_dates([start_date])[0]
                if start_date < valid_start:
                    return None
                end_date = (parser.parse(start_date) + timedelta(hours=23)).isoformat()

            else:
                converted_dates = format_dates(date_range)
                start_date = converted_dates[0]
                end_date = converted_dates[-1]

            reach_ds = nwm_ds[variable_name].sel(feature_id=comid).loc[dict(time=slice(start_date, end_date))]

        df = reach_ds.to_dataframe().reset_index()
        df = df.set_index('time')
        df.index = pd.to_datetime(df.index)

        streamflow = df['streamflow']
        if not streamflow.empty:
            return float(streamflow.mean())


# # tested with same dates, different dates, and no dates... they all work
# same_dates = ['2010-04-12', '2010-04-12']
# dif_dates = ['2010-04-12', '2010-05-12']
# linkno = 760645077
# geo = "GEOGLOWS"
# print(get_streamflow(geo, linkno, same_dates))
# print(get_streamflow(geo, linkno, dif_dates))
# print(get_streamflow(geo, linkno))
#
# reach_id = 368123
# nwm = "National Water Model"
# print(get_streamflow(nwm, reach_id, same_dates))
# print(get_streamflow(nwm, reach_id, dif_dates))
# print(get_streamflow(nwm, reach_id))
