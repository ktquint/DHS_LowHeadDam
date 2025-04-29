"""
This file holds all the objects we'll be using

CrossSections holds all the information we have associated with a given cross-section. The objects will be made using
the ID_XS_Out.txt file for each dam

Dam holds the cross-section information
"""
import ast
import numpy as np
import pandas as pd
import dbfread as dbf
import matplotlib.pyplot as plt


class CrossSection:
    def __init__(self, lhd_id, index, xs_row): # lateral, elevation, wse, distance, location, depth_a, depth_b, lat, lon):
        """
        lateral is a [] of lateral distances
        elevation is a [] of elevations
        wse is water surface elevation as a float
        distance is the downstream or upstream distance from the dam
        location is either 'Downstream' or 'Upstream'

        rating curve eq. d = a * Q **b
        depth_a is a & depth_b is b
        """
        self.id = lhd_id
        # geospatial info
        self.lat = xs_row['Lat']
        self.lon = xs_row['Lon']
        if index == 4:
            self.location = 'Upstream'
        else:
            self.location = 'Downstream'
        # rating curve info
        self.a = xs_row['depth_a']
        self.b = xs_row['depth_b']
        # cross-section plot info
        self.wse = xs_row['wse_1']


        # self.lateral = lateral
        # self.elevation = elevation

        # self.distance = distance


    def trim_to_banks(self):
        # this autofilled... i'll fit it later
        self.lateral = self.lateral[~np.isnan(self.lateral)]
        self.elevation = self.elevation[~np.isnan(self.elevation)]

    def plot_cross_section(self, in_banks=False):
        """
        in_banks is bool.
        """
        if in_banks:
            self.trim_to_banks()

        # cross-section elevations
        plt.plot(self.lateral, self.elevation, color='black')
        # wse line
        # plt.plot(???, self.wse, color='blue')
        plt.xlabel('Lateral Distance (m)')
        plt.ylabel('Elevation (m)')
        plt.title(f'{self.location} Cross-Section {self.distance} meters from LHD No. {self.id}')
        plt.show()

    def create_rating_curve(self, min_Q, max_Q):
        x = np.linspace(min_Q, max_Q, 100)
        y = self.a * x ** self.b
        plt.plot(x, y,
                 label=f'Rating Curve {self.distance} meters {self.location}: $y = {self.a:.3f} x^{{{self.b:.3f}}}$')

    def plot_rating_curve(self, min_Q, max_Q):
        x = np.linspace(min_Q, max_Q, 100)
        y = self.a * x ** self.b
        plt.plot(x, y, color='black', label=f'$y = {self.a:.3f} x^{{{self.b:.3f}}}$')

        plt.xlabel('Flow (m$^{3}$/s')
        plt.ylabel('Depth (m)')
        plt.title(f'{self.location} Rating Curve {self.distance} meters from LHD No. {self.id}')
        plt.legend(title=f'{self.location} Rating Curve Equation')

        plt.grid(True)
        plt.show()


class Dam:
    """
    create a Dam based on the BYU LHD IDs
    add cross-sections with information from vdt & cross_section files
    """
    def __init__(self, lhd_id, lhd_csv, project_dir):
        self.id = lhd_id
        lhd_df = pd.read_csv(lhd_csv)
        id_row = lhd_df[lhd_df['ID'] == self.id]
        self.latitude = id_row['Latitude'].values[0]
        self.longitude = id_row['Longitude'].values[0]
        self.cross_sections = []
        # find attributes based on the vdt and xs files
        vdt_loc = f'{project_dir}/LHD_Results/{self.id}/VDT/{self.id}_Local_CurveFile.dbf'
        xs_loc = f'{project_dir}/LHD_Results/{self.id}/XS/{self.id}_XS_Out.txt'
        # read in dbf then translate it to data.frame
        vdt_dbf = dbf.DBF(vdt_loc)
        vdt_df = pd.DataFrame(iter(vdt_dbf))
        # read in txt as data.frame
        xs_df = pd.read_csv(xs_loc, header=None, sep='\t')
        xs_df.rename(columns={0: 'COMID', 1: 'Row', 2: 'Col',
                              3: 'elev_1', 4: 'wse_1', 5: 'lat_1', 6: 'n_1',
                              7: 'elev_2', 8: 'wse_2', 9: 'lat_2', 10: 'n_2'},
                     inplace=True)
        # evaluate the strings as literals (lists)
        xs_df['elev_1'] = xs_df['elev_1'].apply(ast.literal_eval)
        xs_df['n_1'] = xs_df['n_1'].apply(ast.literal_eval)
        xs_df['elev_2'] = xs_df['elev_2'].apply(ast.literal_eval)
        xs_df['n_2'] = xs_df['n_2'].apply(ast.literal_eval)
        # let's merge the tables
        merged_df = pd.merge(vdt_df, xs_df, on=['Row', 'Col'], how='left')
        # let's go through each row of the df and add cross-sections to the dam.
        for index, row in merged_df.iterrows():
            self.cross_sections.append(CrossSection(lhd_id, index, row))

    def plot_rating_curves(self):
        for cross_section in self.cross_sections:
            cross_section.plot_rating_curve()
            plt.xlabel('Flow (m$^{3}$/s')
            plt.ylabel('Depth (m)')
            plt.title(f'Rating Curves for LHD No. {self.id}')
            plt.show()

    def plot_cross_sections(self):
        for cross_section in self.cross_sections:
            cross_section.plot_cross_section()
