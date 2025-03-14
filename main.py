import ast
import pandas as pd
import dbfread as dbf
import re

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
