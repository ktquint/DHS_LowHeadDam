import os
from classes import Dam


# project_dir holds all the project files
project_dir = "C:/Users/ki87ujmn/Downloads/LHD_RathCelon"
# results_dir holds the ARC runs
results_dir = os.path.join(project_dir, "LHD_Results")
# search for the only csv file and save its path
lhd_csv = [f for f in os.listdir(project_dir) if f.endswith('.csv')][1]
csv_path = os.path.join(project_dir, lhd_csv)


# get the path of all folders in results, trim them to the dam id and then turn the dam ids into strings
dam_paths = [os.path.join(results_dir, d) for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
dam_strs = [os.path.basename(path.rstrip(os.sep)) for path in dam_paths]
dam_ints = [int(x) for x in dam_strs]

for dam_id in dam_ints:

    if dam_id == 38:
        print(f"Analyzing Dam No. {dam_id}")
        dam_i = Dam(int(dam_id), csv_path, project_dir)
        dam_i.plot_rating_curves()
        dam_i.plot_cross_sections()
        # dam_i.plot_all_curves()
        print("Onto the next one! :)")
        # dam_i.plot_map()

