import os
import pandas as pd
import create_json as cj, download_dem as dd, download_stream as ds


"""
this is where the magic happens...
"""

# this is the database I'm working with:
project_folder = "C:/Users/ki87ujmn/Downloads/LHD_RathCelon"
lhd_database = project_folder + "/LowHead_Dam_Database.xlsx"
# these are where I'll store the DEMs and stream geopackages
dem_folder = project_folder + "/LHD_DEMs"
os.makedirs(dem_folder, exist_ok=True)
strm_folder = project_folder + "/LHD_STRMs"
os.makedirs(strm_folder, exist_ok=True)
# this is where I'll store the RathCelon output
results_folder = project_folder + "/LHD_Results"
os.makedirs(results_folder, exist_ok=True)
# we'll turn the finished data_frame into a csv with the name:
lhd_csv = project_folder + "/LowHead_Dam_Database.csv"

# convert your database to a data_frame
lhd_df = pd.read_excel(lhd_database)
lhd_dem = dd.download_dems(lhd_df, dem_folder) # this function should take a dataframe, output folder (where the DEMs are downloaded) and return a dataframe
# now that we have DEMs, let's get the streamlines
lhd_strm = ds.assign_flowlines(lhd_dem, strm_folder) # this should take a dataframe, output folder (where the geopackages are downloaded) and return a dataframe
# add a column with the location of the results folder
lhd_strm['output_dir'] = results_folder
# a little convoluted, but we need a .csv for rathcelon to read
lhd_strm.to_csv(lhd_csv, index=False)
input_loc = cj.rathcelon_input(lhd_csv, project_folder)
# now type open the terminal in the rathcelon repository and type `rathcelon json ` + the input_loc path
