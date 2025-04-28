"""
This file holds all the objects we'll be using

CrossSections holds all the information we have associated with a given cross-section. The objects will be made using
the ID_XS_Out.txt file for each dam

Dam holds the cross-section information
"""
import numpy as np
import pandas as pd
import dbfread as dbf
import matplotlib.pyplot as plt


class CrossSection:
    def __init__(self, lhd_id, lateral, elevation, wse, distance, location, depth_a, depth_b):
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
        self.lateral = lateral
        self.elevation = elevation
        self.wse = wse
        self.distance = distance
        self.location = location
        self.a = depth_a
        self.b = depth_b

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
        vdt_loc = f'{project_dir}/LHD_Results/{self.id}/VDT/{id}_Local_CurveFile.dbf'
        xs_loc = f'{project_dir}/LHD_Results/{self.id}/XS/{id}_XS_Out.txt'


        self.cross_sections = []

    def add_cross_section(self, vdt_file, cross_section_file):
        """
        vdt_file is the ID_Local_CurveFile.dbf file
        cross_section_file is the ID_XS_Out.txt file
        """
        attribute_table = dbf.DBF(vdt_file)


        self.cross_sections.append(cross_section)

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
