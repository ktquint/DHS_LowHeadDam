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
from scipy.optimize import fsolve

g = 9.81 # grav. const.


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
        self.max_Q = xs_row['Q_max']
        # cross-section plot info
        self.wse = xs_row['wse_1']
        y_1 = xs_row['elev_1']
        y_1 = y_1[::-1]
        y_2 = xs_row['elev_2']
        self.elevation = y_1 + y_2
        x_1 = [0 + j * xs_row['lat_1'] for j in range(len(y_1))]
        x_2 = [max(x_1) + j * xs_row['lat_2'] for j in range(len(y_2))]
        self.lateral = x_1 + x_2
        # fix this later...
        self.distance = 100


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

    def create_rating_curve(self):
        x = np.linspace(0, self.max_Q, 100)
        y = self.a * x ** self.b
        plt.plot(x, y,
                 label=f'Rating Curve {self.distance} meters {self.location}: $y = {self.a:.3f} x^{{{self.b:.3f}}}$')

    def plot_rating_curve(self):
        x = np.linspace(0, self.max_Q, 100)
        y = self.a * x ** self.b
        plt.plot(x, y, color='black', label=f'$y = {self.a:.3f} x^{{{self.b:.3f}}}$')

        plt.xlabel('Flow (m$^{3}$/s')
        plt.ylabel('Depth (m)')
        plt.title(f'{self.location} Rating Curve {self.distance} meters from LHD No. {self.id}')
        plt.legend(title=f'{self.location} Rating Curve Equation')

        plt.grid(True)
        plt.show()

    # def plot_


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
        self.top_width = 0
        self.weir_length = 0
        self.height = 0
        self.h_overtop = []
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
        self.max_Q = max(merged_df['QMax'].values)
        # let's go through each row of the df and add cross-sections to the dam.
        for index, row in merged_df.iterrows():
            self.cross_sections.append(CrossSection(lhd_id, index, row))

    def set_dam_height(self):
        self.height = 4

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

    def plot_flip_depth(self):
        Q_array = np.linspace(0, self.max_Q, 100)
        A = (2 * self.top_width / 3) * np.sqrt(2 * g)
        H_list = []
        for Q in Q_array:
            H_guess: float = 1.0
            H_sol = fsolve(self.weir_head, H_guess, args=(Q,A))
            H_list.append(H_sol[0])
        H_array = np.array(H_list)
        self.h_overtop = H_array
        y_flip = (H_array + self.height) / 1.1
        plt.plot(Q_array, y_flip, label=f'Flip Depth (m)')

    def plot_seq_depth(self):
        Q_array = np.linspace(0, self.max_Q, 100)
        y_1_list = []
        for H in self.h_overtop:
            x_0 = 0.5
            x_sol = fsolve(self.weir_head, x_0, args=(H,))[0]
            y_1 = x_sol * H
            y_1_list.append(y_1)
        y_1_array = np.array(y_1_list)

        plt.plot(Q_array, y_1_array, label=f'Sequent Depth (m)')

    def weir_head(self, H, Q, A):
        return 0.611 * H**(3/2) + (0.075 / self.height) * H**(5/2) - Q/A
    
    def y1_over_H(self, x, H):
        term_1 = x**3
        term_2 = (1 + self.height / H) * x**2
        term_3 = (4/9) * (0.611 + 0.075 * H / self.height)**2 + (1 + 0.1 * self.height/H)
        return term_1 - term_2 + term_3
        