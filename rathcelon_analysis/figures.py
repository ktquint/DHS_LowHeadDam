import re
import ast
import pandas as pd
import dbfread as dbf
import matplotlib.pyplot as plt

"""
these functions extract data frames from raw files
"""

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


def get_within_banks(xs_df):
    """
    this doesn't work yet... lol
    """
    # this is the manning's for flow within banks
    mannings_n = 0.03
    for i in range(len(xs_df)):
        index_bank1 = len(xs_df.loc[i, "manning's_n1"]) - 1 - xs_df.loc[i, "manning's_n1"][::-1].index(mannings_n)
        xs_df.loc[i, 'xs_profile1'] = xs_df.loc[i, 'xs_profile1'][:index_bank1]
        xs_df.loc[i, "manning's_n1"] = xs_df.loc[i, "manning's_n1"][:index_bank1]
        index_bank2 = len(xs_df.loc[i, "manning's_n2"]) - 1 - xs_df.loc[i, "manning's_n2"][::-1].index(mannings_n)
        xs_df.loc[i, 'xs_profile2'] = xs_df.loc[i, 'xs_profile2'][:index_bank2]
        xs_df.loc[i, "manning's_n2"] = xs_df.loc[i, "manning's_n2"][:index_bank2]
    return xs_df


def plot_cross_sections(attribute_df, xs_df, output_dir, in_banks):
    # this id will be used to name the output files
    lhd_id = attribute_df['lhd_id'][0]

    # if we're only plotting in banks we need to trim down the xs data frame
    if in_banks:
        xs_df = get_within_banks(xs_df)

    xs_profile = pd.DataFrame()
    for i in range(len(attribute_df)):
        result = xs_df[(xs_df['row'] == attribute_df['row'][i]) & (xs_df['column'] == attribute_df['col'][i])]
        xs_profile = pd.concat([xs_profile, result])

    for i in range(len(xs_profile)):
        # this is the cross-section elevations
        y1 = xs_profile['xs_profile1'].iloc[i]
        # they're backwards, so let's reverse them
        y1 = y1[::-1]
        # now we'll read in the second cross-section and combine lists
        y2 = xs_profile['xs_profile2'].iloc[i]
        y = y1 + y2
        # let's make a list of horizontal distances based on the z distance
        x1 = [0 + j * xs_profile['d_distance_z1'].iloc[i] for j in range(len(y1))]
        x2 = [max(x1) + j * xs_profile['d_distance_z2'].iloc[i] for j in range(len(y2))]
        x = x1 + x2

        # create the plot
        plt.plot(x, y)

        # add labels and title
        plt.xlabel('Lateral Distance (m)')
        plt.ylabel('Elevation (m)')
        plt.title(f'Cross-section No. {i + 1} at LHD No. {lhd_id}')

        # display plot
        png_output = output_dir + f'/Cross-section No.{i+1} at LHD No.{lhd_id}.png'
        plt.savefig(png_output, dpi=300, bbox_inches='tight')
        plt.show()

"""
Test Case: 
"""

test_dbf = "C:/Users/ki87ujmn/Downloads/rathcelon-example/results/28/VDT/28_Local_CurveFile.dbf"
test_txt = "C:/Users/ki87ujmn/Downloads/rathcelon-example/results/28/XS/28_XS_Out.txt"
test_output = 'C:/Users/ki87ujmn/Downloads'

test_att_tbl = get_attribute_df(test_dbf)
test_df = get_xs_df(test_txt)

plot_rating_curve(test_att_tbl, test_output)
plot_cross_sections(test_att_tbl, test_df, test_output, False)
