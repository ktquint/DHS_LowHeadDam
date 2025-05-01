"""
This file holds all the objects we'll be using

CrossSections holds all the information we have associated with a given cross-section. The objects will be made using
the ID_XS_Out.txt file for each dam

Dam holds the cross-section information
"""
import ast
import math
import geoglows
import numpy as np
import pandas as pd
import dbfread as dbf
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import requests
import re

g = 9.81 # grav. const.

def round_sigfig(num, sig_figs):
    if num == 0:
        return 0
    else:
        return round(num, sig_figs - int(math.floor(math.log10(abs(num)))) - 1)


def get_streamflow(comid, date_range=None):
    """
    comid needs to be an int
    date needs to be in the format %Y-%m-%d

    returns average streamflow—for the entire record if no date is given, else it's the average on the given date
    """
    try:
        comid = int(comid)
    except ValueError:
        raise ValueError("comid needs to be an int")

    # this is all the data for the comid
    historic_df = geoglows.data.retro_daily(comid)
    historic_df.index = pd.to_datetime(historic_df.index)

    if date_range is not None:
        try:
            subset_df = historic_df.loc[date_range[0]:date_range[1]]
            Q = np.median(subset_df[comid])
        except IndexError:
            raise ValueError(f"NO data available for {date_range}")
    else:
        Q = np.median(historic_df[comid])

    return Q

def get_dem_dates(lat, lon):
    """
    Use lat/lon to get Lidar data used to make the DEM.
    Check the date the Lidar was taken.
    """
    bbox = (lon - 0.001, lat - 0.001, lon + 0.001, lat + 0.001)
    dataset = "Lidar Point Cloud (LPC)"
    base_url = "https://tnmaccess.nationalmap.gov/api/v1/products"

    params = {
        "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
        "datasets": dataset,
        "max": 1,
        "outputFormat": "JSON"
    }
    response = requests.get(base_url, params=params)

    lidar_info = response.json().get("items", [])
    if not lidar_info:
        return "No Lidar data found for the given coordinates."

    meta_url = lidar_info[0].get('metaUrl')
    if not meta_url:
        return "metaUrl key not found in the response."

    response2 = requests.get(meta_url)
    html_content = response2.text

    match_start = re.search(r'<dt>Start Date</dt>\s*<dd>(.*?)</dd>', html_content, re.IGNORECASE)
    match_end = re.search(r'<dt>End Date</dt>\s*<dd>(.*?)</dd>', html_content, re.IGNORECASE)

    if match_start and match_end:
        start_date_value = match_start.group(1).strip()
        end_date_value = match_end.group(1).strip()
        return f"Start Date: {start_date_value}, End Date: {end_date_value}"
    else:
        return "Date parameters not found."

def weir_height(Q, b, y_u, tol=0.001):
    """
    Q = flow in river (cms)
    b = bank width (m)
    y_u = upstream depth (m)
    """
    q = Q / b # unit flow
    # left-hand side
    A = 3 * q / (2 * np.sqrt(2 * g))
    # initial weir height estimate
    p = 0.5 * y_u # we want to start with a positive number < y_u
    # right-hand side
    B = 0.611 * (y_u - p)**(3/2) + 0.075 * ((y_u - p)**(5/2))/p
    counter = 0 # to avoid infinite loop
    while abs(A - B) > tol:
        counter += 1
        if A < B:
            p += 0.001
        else:
            p -= 0.001
        # recalculate B after adjusting height
        B = 0.611 * (y_u - p) ** (3 / 2) + 0.075 * ((y_u - p) ** (5 / 2)) / p
        if counter > 10000:
            break
    return round(p, 3)


class CrossSection:
    def __init__(self, lhd_id, index, xs_row):
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
        self.max_Q = xs_row['QMax']
        self.slope = round_sigfig(xs_row['slope'], 3)
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
        plt.plot(self.lateral, self.elevation,
                 color='black', label=f'Downstream Slope: {self.slope}')
        # wse line
        # plt.plot(???, self.wse, color='blue')
        plt.xlabel('Lateral Distance (m)')
        plt.ylabel('Elevation (m)')
        plt.title(f'{self.location} Cross-Section {self.distance} meters from LHD No. {self.id}')
        plt.legend(loc='upper left')
        plt.show()


    def create_rating_curve(self):
        x = np.linspace(1, self.max_Q, 100)
        y = self.a * x ** self.b
        plt.plot(x, y,
                 label=f'Rating Curve {self.distance} meters {self.location}: $y = {self.a:.3f} x^{{{self.b:.3f}}}$')


    def plot_rating_curve(self):
        x = np.linspace(1, self.max_Q, 100)
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
        #database information
        self.id = lhd_id
        lhd_df = pd.read_csv(lhd_csv)
        id_row = lhd_df[lhd_df['ID'] == self.id]

        # geographic information
        self.latitude = id_row['latitude'].values[0]
        self.longitude = id_row['longitude'].values[0]

        # physical information
        self.cross_sections = []
        self.weir_length = id_row['weir_length'].values[0]

        # self.top_width = 0
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
        xs_df.rename(columns={0: 'COMID', 1: 'Row', 2: 'Col', 3: 'elev_1',
                              4: 'wse_1', 5: 'lat_1', 6: 'n_1', 7: 'elev_2',
                              8: 'wse_2', 9: 'lat_2', 10: 'n_2', 11: 'slope'},
                     inplace=True)
        # evaluate the strings as literals (lists)
        xs_df['elev_1'] = xs_df['elev_1'].apply(ast.literal_eval)
        xs_df['n_1'] = xs_df['n_1'].apply(ast.literal_eval)
        xs_df['elev_2'] = xs_df['elev_2'].apply(ast.literal_eval)
        xs_df['n_2'] = xs_df['n_2'].apply(ast.literal_eval)
        # let's merge the tables (how='left' because vdt only contains the xs we want and xs has all of them)
        merged_df = pd.merge(vdt_df, xs_df, on=['COMID', 'Row', 'Col'], how='left')
        self.max_Q = max(merged_df['QMax'].values)
        # let's go through each row of the df and add cross-sections to the dam.
        for index, row in merged_df.iterrows():
            self.cross_sections.append(CrossSection(lhd_id, index, row))

        # hydrologic information
        self.comid = merged_df['COMID'].values[-1] # get the comid at the upstream cross-section
        # now we'll define the baseflow for the DEM
        self.dem_dates = get_dem_dates(self.latitude, self.longitude)

        self.dem_dates = [id_row['dem_start'].values[0], id_row['dem_end'].values[0]]
        if not self.dem_dates: # if there is no date, we'll take the median flow
            self.dem_flow = get_streamflow(self.comid)
        else:
            self.dem_flow = get_streamflow(self.comid, self.dem_dates)
        # then we'll find the flow on the day of the fatality
        self.fatality_date = id_row['fatality_date'].values[0]
        self.fatality_flow = get_streamflow(self.comid, [self.fatality_date, self.fatality_date])
        # self.Q_median = get_streamflow(self.comid)

        # let's add the dam height here

        for cross_section in self.cross_sections[:-1]:
            energy_up = self.cross_sections[-1].wse - min(cross_section.elevation)
            print("weir height estimate: ")
            P_guess = weir_height(54.516042, self.weir_length, energy_up)
            print(P_guess * 3.281)


    def set_dam_height(self, P):
        self.height = P


    def plot_rating_curves(self):
        for cross_section in self.cross_sections:
            cross_section.create_rating_curve()
        plt.xlabel('Flow (m$^{3}$/s')
        plt.ylabel('Depth (m)')
        plt.title(f'Rating Curves for LHD No. {self.id}')
        plt.legend(title="Rating Curve Equations", loc='best', fontsize='small')
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
        