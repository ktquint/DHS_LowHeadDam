import numpy as np
import matplotlib.pyplot as plt
from main import get_attribute_df


def plot_rating_curve(attribute_df, output_dir):
    lhd_id = attribute_df['lhd_id'][0]
    # define range of flows
    x = np.linspace(1, 100, 100)

    # initialize the plot
    plt.figure(figsize=(10, 6))

    # iterate through the rating curves
    for index, row in attribute_df.iterrows():
        a = row['depth_a']
        b = row['depth_b']
        y = a * x**b
        plt.plot(x, y, label=f'Downstream Cross-section {row["id"]}: $y = {a:.3f} x^{{{b:.3f}}}$')

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
test_dbf = "C:/Users/ki87ujmn/Downloads/rathcelon-example/results/272/VDT/272_Local_CurveFile.dbf"
test_output = 'C:/Users/ki87ujmn/Downloads'
test_att_tbl = get_attribute_df(test_dbf)
plot_rating_curve(test_att_tbl, test_output)
