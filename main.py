import re
import ast
import pandas as pd
import dbfread as dbf
import create_json as cj
import download_dem as dd
import download_gpkgs as ds # ds for download stream


def get_attribute_df(curve_dbf):
    # create attribute table based on .dbf file
    attribute_table = dbf.DBF(curve_dbf)

    # create id, row, col, depth_a, and depth_b lists
    id_list = list(range(1, len(attribute_table) + 1))
    row_list, col_list, depth_a, depth_b = [], [], [], []
    lhd_id_list = [re.sub(r'\D', '', curve_dbf[-23:-20])] * len(id_list)

    for attribute in attribute_table:
        row_list.append(attribute["Row"])
        col_list.append(attribute["Col"])
        depth_a.append(attribute["depth_a"])
        depth_b.append(attribute["depth_b"])

    # convert rating curve equations into a dataframe
    attribute_df = pd.DataFrame({'id': id_list,
                                 'row': row_list, 'col': col_list,
                                 'depth_a': depth_a, 'depth_b': depth_b,
                                 'lhd_id': lhd_id_list})
    return attribute_df


def get_xs_df(xs_txt):
    xs_df = pd.read_csv(xs_txt, header=None, sep='\t')
    xs_df = xs_df.rename(columns={0: 'cell_comid', 1: 'row', 2: 'column',
                                  3: 'xs_profile1', 4: 'd_wse', 5: 'd_distance_z1', 6: "manning's_n1",
                                  7: 'xs_profile2', 8: 'd_wse', 9: 'd_distance_z2', 10: "manning's_n2"})
    xs_df['xs_profile1'] = xs_df['xs_profile1'].apply(ast.literal_eval)
    xs_df['xs_profile2'] = xs_df['xs_profile2'].apply(ast.literal_eval)
    xs_df["manning's_n1"] = xs_df["manning's_n1"].apply(ast.literal_eval)
    xs_df["manning's_n2"] = xs_df["manning's_n2"].apply(ast.literal_eval)
    return xs_df

"""
this is where the magic happens...
"""

# this is the database I'm working with:
project_folder = "C:/Users/ki87ujmn/Downloads/LHD_RathCelon"
lhd_database = project_folder + "/LowHead_Dam_Database.xlsx"
# these are where I'll store the DEMs and stream geopackages
dem_folder = project_folder + "/LHD_DEMs"
strm_folder = project_folder + "/LHD_STRMs"
# this is where I'll store the RathCelon output
results_folder = project_folder + "/LHD_Results"
# we'll turn the finished data_frame into a csv with the name:
lhd_csv = project_folder + "/LowHead_Dam_Database.csv"

# convert your database to a data_frame
lhd_df = pd.read_excel(lhd_database)
lhd_dem = dd.search_and_download_dems(lhd_df, dem_folder) # this function should take a dataframe, output folder (where the DEMs are downloaded) and return a dataframe
# now that we have DEMs, let's get the streamlines
lhd_strm = ds.gpkg_download(lhd_dem, strm_folder) # this should take a dataframe, output folder (where the geopackages are downloaded) and return a dataframe
# add a column with the location of the results folder
lhd_strm['output_dir'] = results_folder
# a little convoluted, but we need a .csv for rathcelon to read
lhd_strm.to_csv(lhd_csv, index=False)
input_loc = cj.rathcelon_input(lhd_csv, project_folder)
# now type open the terminal in the rathcelon repository and type 'rathcelon json ' + the input_loc path
