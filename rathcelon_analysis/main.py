import os
from classes import Dam

"""
if you're going to have a separate main
you need to figure out what to do about
lhd_id. e.g., make a new one in each of
the functions based on the file name or
something like that.
"""


# this folder has the results from some example runs
project_dir = "C:/Users/ki87ujmn/Downloads/LHD_RathCelon"
results_dir = project_dir + "/LHD_Results"
lhd_csv = project_dir + "/LowHead_Dam_Database.csv"
# these are the subdirectories for each rathcelon run
dams = [os.path.join(results_dir, d) for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]

for dam in dams:
    # the lhd_id is the name of each directory
    lhd_id = dam.split('\\')[-1] # this is for a PC
    dam_i = Dam(int(lhd_id), lhd_csv, project_dir)

    dam_i.plot_rating_curves()
    dam_i.plot_cross_sections()
