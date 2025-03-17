import json
import pandas as pd
from typing import Any


def rathcelon_input (lhd_csv, output_loc):
    lhd_df = pd.read_csv(lhd_csv)
    dams: list[dict[str | Any, str | bool | Any]] = []
    for index, row in lhd_df.iterrows():
        name = str(row["ID"])
        dam_csv = lhd_csv
        dam_id_field = "ID"
        dam_id = row["ID"]
        flowline = row["flowline"]
        dem_dir = row["dem_dir"]
        output_dir = row["output_dir"]
        dam_dict = {
            "name" : name,
            "dam_csv" : dam_csv,
            "dam_id_field" : dam_id_field,
            "dam_id" : dam_id,
            "flowline" : flowline,
            "dem_dir" : dem_dir,
            "bathy_use_banks": False,
            "output_dir" : output_dir,
            "process_stream_network" : True,
            "find_banks_based_on_landcover" : False,
            "create_reach_average_curve_file" : False
            }
        dams.append(dam_dict)
    input_data = {
        "dams" : dams
    }
    file_name = output_loc + "/input.json"
    with open(file_name, 'w') as json_file:
        json.dump(input_data, json_file, indent=4)

test_csv = "C:/Users/adele/Downloads/Low head Dam Info - Copy for python(Slopes).csv"
output_fold = "C:/Users/adele/Downloads"
rathcelon_input(test_csv, output_fold)
