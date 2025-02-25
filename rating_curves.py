import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import dbfread as dbf

def plot_rating_curve(dbf_file, output_dir):
    # grad the LHD ID number
    lhd_id = dbf_file[-23:-20]

    # create attribute table based on .dbf file
    attribute_table = dbf.DBF(dbf_file)

    # create id, depth_a, and depth_b lists
    id_list = list(range(1, len(attribute_table)+1))
    depth_a = []
    depth_b = []
    for attribute in attribute_table:
        depth_a.append(attribute["depth_a"])
        depth_b.append(attribute["depth_b"])

    # convert rating curve equations into a dataframe
    rating_curve = pd.DataFrame({'id': id_list, 'depth_a': depth_a, 'depth_b': depth_b})

    # define range of flows
    x = np.linspace(1, 100, 100)

    # initialize the plot
    plt.figure(figsize=(10, 6))

    # iterate through the rating curves
    for index, row in rating_curve.iterrows():
        a = row['depth_a']
        b = row['depth_b']
        y = a * x**b
        plt.plot(x, y, label=f'Downstream Cross-section {row["id"]}: $y = {a:.3f} \cdot x^{{{b:.3f}}}$')

    # Add labels and legend
    plt.xlabel('Flow (m$^{3}$/s)')
    plt.ylabel('Depth (m)')
    plt.title(f'Downstream Rating Curves at LHD No. {lhd_id}')
    plt.legend(title="Rating Curve Equations", loc='best', fontsize='small')  # Add legend
    plt.grid(True)

    # create output path for png file and save it
    png_output = output_dir + '/' + lhd_id + '_Rating_Curves.png'
    plt.savefig(png_output, dpi=300, bbox_inches='tight')
    plt.show()

"""
Test Case: 
"""
# test_dbf = "C:/Users/ki87ujmn/Downloads/rathcelon-example/results/272/VDT/272_Local_CurveFile.dbf"
# test_output = 'C:/Users/ki87ujmn/Downloads'
# plot_rating_curve(test_dbf, test_output)
