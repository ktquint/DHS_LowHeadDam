import pandas as pd
import json
import matplotlib.pyplot as plt
from main import get_attribute_df

def get_within_banks(xs_df):


def plot_cross_section(attribute_df, xs_txt, output_dir):
    xs_df = pd.read_csv(xs_txt, header=None, sep='\t')
    xs_df = xs_df.rename(columns={0: 'cell_comid', 1: 'row', 2: 'column',
                                  3: 'xs_profile1', 4: 'd_wse', 5: 'd_distance_z1', 6: 'mannings_n1',
                                  7: 'xs_profile2', 8: 'd_wse', 9: 'd_distance_z2', 10: 'mannings_n2'})
    xs_profile = pd.DataFrame()
    for i in range(len(attribute_df)):
        result = xs_df[(xs_df['row'] == attribute_df['row'][i]) & (xs_df['column'] == attribute_df['col'][i])]
        xs_profile = pd.concat([xs_profile, result])

    for i in range(len(xs_profile)):
        # this is the cross-section elevations
        y1 = json.loads(xs_profile['xs_profile1'].iloc[i])
        # they're backwards, so let's reverse them
        y1 = y1[::-1]
        # now we'll read in the second cross-section and combine lists
        y2 = json.loads(xs_profile['xs_profile2'].iloc[i])
        y = y1 + y2
        # let's make a list of horizontal distances based on the z distance
        x = [0 + j * xs_profile['d_distance_z1'].iloc[i] for j in range(len(y))]

        # create the plot
        plt.plot(x, y)

        # add labels and title
        plt.xlabel('Width (m)')
        plt.ylabel('Height (m)')
        plt.title(f'Cross-section No. {i + 1}')

        # display plot
        plt.show()

"""
Test Case: 
"""
test_dbf = "C:/Users/ki87ujmn/Downloads/rathcelon-example/results/272/VDT/272_Local_CurveFile.dbf"
test_txt = "C:/Users/ki87ujmn/Downloads/rathcelon-example/results/272\XS/272_XS_Out.txt"
test_output = 'C:/Users/ki87ujmn/Downloads'
test_att_tbl = get_attribute_df(test_dbf)
plot_cross_section(test_att_tbl, test_txt, test_output)
