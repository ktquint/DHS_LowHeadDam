import json
import pandas as pd


def rathcelon_input(lhd_csv, output_loc, nwm_parquet=None):
    lhd_df = pd.read_csv(lhd_csv)
    dams = []
    for index, row in lhd_df.iterrows():
        name = str(row["ID"])
        dam_csv = lhd_csv
        dam_id_field = "ID"
        dam_id = row["ID"]
        hydrography = row["hydrography"]
        hydrology = row["hydrology"]


        if hydrography == "GEOGLOWS":
            flowline = row["flowline_TDX"]
        else:
            flowline = row["flowline_NHD"]

        res_order = ['dem_1m', 'dem_3m', 'dem_10m']
        dem_dir = None
        for res in res_order:
            val = row[res]
            if pd.notna(val):
                dem_dir = val
                break

        output_dir = row["output_dir"]

        if hydrology == "GEOGLOWS":
            streamflow = row["flowline_TDX"]
            known_baseflow = row['dem_baseflow_GEOGLOWS']
        else:
            streamflow = nwm_parquet
            known_baseflow = row['dem_baseflow_NWM']

        # Check if known_baseflow is a string or NaN (which is not valid JSON)
        # and convert it to None (which becomes 'null' in JSON)
        if isinstance(known_baseflow, str) or pd.isna(known_baseflow):
            known_baseflow = None

        dam_dict = {"name": name,
                    "dam_csv": dam_csv,
                    "dam_id_field": dam_id_field,
                    "dam_id": dam_id,
                    "flowline": flowline,
                    "dem_dir": dem_dir,
                    "bathy_use_banks": False,
                    "output_dir": output_dir,
                    "process_stream_network": True,
                    "find_banks_based_on_landcover": False,
                    "create_reach_average_curve_file": False,
                    "known_baseflow": known_baseflow,
                    "streamflow": streamflow}
        dams.append(dam_dict)

    input_data = {"dams": dams}
    with open(output_loc, 'w') as json_file:
        # noinspection PyTypeChecker
        json.dump(input_data, json_file, indent=4)

    # this will tell us where we saved the input file

    return dams