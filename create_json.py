import json
from typing import Any

import pandas as pd

# Data to be written to the JSON file
data = {
    "dams" :
        []
}

def rathclon_input (lhd_csv, outputlocation): #created function, rathclon_input is function name, () = function peramiters
    lhd_df = pd.read_csv(lhd_csv) #creates a dataframe out of CSV
    dams: list[dict[str | Any, str | bool | Any]] = [] #says which data types go into "dams" list
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
    data = {
        "dams" : dams
    }
    file_name = "data.json"
    with open(file_name, 'w') as json_file:
        json.dump(data, json_file, indent=4)

test_csv = "C:/Users/adele/Downloads/Low head Dam Info - Copy for python(Slopes).csv"
output_fold = "C:/Users/adele/Downloads"
rathclon_input (test_csv, output_fold)