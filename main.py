import pandas as pd
import dbfread as dbf

def get_attribute_df(dbf_file):
    # create attribute table based on .dbf file
    attribute_table = dbf.DBF(dbf_file)

    # create id, row, col, depth_a, and depth_b lists
    id_list = list(range(1, len(attribute_table) + 1))
    row_list, col_list, depth_a, depth_b = [], [], [], []
    lhd_id_list = [dbf_file[-23:-20]] * len(id_list)

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
