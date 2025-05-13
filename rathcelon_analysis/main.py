import os
from classes import Dam


# this folder has the results from some example runs
project_dir = "C:/Users/ki87ujmn/Downloads/rathcelon-example"
results_dir = project_dir + "/LHD_Results"
lhd_csv = project_dir + "/LHD_Lat_Long.csv"
# these are the subdirectories for each rathcelon run
dams = [os.path.join(results_dir, d) for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]

for dam in dams:
    # the lhd_id is the name of each directory
    lhd_id = dam.split('\\')[-1] # this is for a PC
    dam_i = Dam(int(lhd_id), lhd_csv, project_dir)

    # dam_i.plot_rating_curves()
    # dam_i.plot_cross_sections()
    # dam_i.plot_all_curves()
    dam_i.plot_map()
