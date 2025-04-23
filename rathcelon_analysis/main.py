import os
import figures as fig

"""
if you're going to have a separate main
you need to figure out what to do about
lhd_id. e.g., make a new one in each of
the functions based on the file name or
something like that.
"""


# this folder has the results from some example runs
results_dir = "C:/Users/ki87ujmn/Downloads/LHD_RathCelon/LHD_Results"
# these are the subdirectories for each rathcelon run
dams = [os.path.join(results_dir, d) for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]

for dam in dams:
    # the lhd_id is the name of each directory
    lhd_id = dam.split('\\')[-1] # this is for a PC
    # the dbf and txt will be in the same place for each file
    result_dbf = dam + f'/VDT/{lhd_id}_Local_CurveFile.dbf'
    result_txt = dam + f'/XS/{lhd_id}_XS_Out.txt'
    # get the attribute table from the dbf
    attr_tbl = fig.get_attribute_df(result_dbf)
    # get the cross-section data from the txt
    cross_section = fig.get_xs_df(result_txt)
    # plot rating curves and cross-sections
    fig.plot_rating_curve(attr_tbl, dam)
    fig.plot_cross_sections(attr_tbl, cross_section, dam, False)
