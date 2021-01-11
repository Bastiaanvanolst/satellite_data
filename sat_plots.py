"""
Plotting and analysis of the DISCOS data retrieved by sat_data.py

Usage example;
import sat_plots
p = sat_plots.SatPlots()
p......
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import pandas as pd
import numpy as np
import math
import datetime
import re
import ast

import sat_data

# TODO understand missing values in datasets. Payloads without launch id? Government or commercial? Increasing occurrence?
# TODO add handling of undetermined orbit class
# TODO consider decay_year - launch_year as a measurement of lifetime.

class SatPlots(object):
    def __init__(self):
        self.G = 6.67430e-11 # Gravitational constant (m,kg,s)
        self.M_earth = 5.972e24 # Earth's mass (kg)
        self.R_earth = 6371000 # Earth's radius (m)

        self.df_frags = sat_data.get_data(database = 'fragmentations')
        self.df_fragevents = sat_data.get_data(database = 'fragmentation-event-types')
        df = sat_data.get_data(database = 'objects')
        # convert values stored as list (read_csv imports them as string)
        df['InitOrbitId'] = df.InitOrbitId.apply(lambda x: np.nan if x is np.nan else list(ast.literal_eval(x))) 
        df['DestOrbitId'] = df.DestOrbitId.apply(lambda x: np.nan if x is np.nan else list(ast.literal_eval(x)))   
        df = self.assign_fragmentationid(df)
        self.df = df
        self.df_launches = sat_data.get_data(database = 'launches')
        self.df_reentries = sat_data.get_data(database = 'reentries')
        self.df_launchsites = sat_data.get_data(database = 'launch-sites')
        self.df_launchsystems = sat_data.get_data(database = 'launch-systems')
        self.df_launchvehicles = sat_data.get_data(database = 'launch-vehicles')
        self.df_propellants = sat_data.get_data(database = 'propellants')
        self.df_entities = sat_data.get_data(database = 'entities')
        self.df_initorbits = sat_data.get_data(database = 'initial-orbits')
        #self.df_initorbits = self.assign_orbit_class(df_initorbits)
        self.df_destorbits = sat_data.get_data(database = 'destination-orbits')
        #self.df_destorbits = self.assign_orbit_class(df_destorbits)
        self.df_ucs = sat_data.get_ucsdata()

        self.junk_obj = ['Rocket Body',
                    'Unknown',
                    'Rocket Mission Related Object',
                    'Payload Mission Related Object',
                    'Rocket Fragmentation Debris',
                    'Payload Fragmentation Debris',
                    'Payload Debris',
                    'Rocket Debris',
                    'Other Debris',
                    'Other Mission Related Object']


    def plot_orbit_density(self, yr = 2020):
        """
        Plots the number density of satellites for a given year, as a function
        of SemiMajorAxis and Inclination
        """
        df_io = self.df_initorbits
        df_do = self.df_destorbits
        df = self.select_pop_in_year(yr = yr)

        df['InitOrbitEpoch'] = df.InitOrbitId.apply(lambda x: np.nan if x is np.nan else np.array(df_io[df_io.OrbitId.isin(x)].Epoch.dt.date))
        df['DestOrbitEpoch'] = df.DestOrbitId.apply(lambda x: np.nan if x is np.nan else np.array(df_do[df_do.OrbitId.isin(x)].Epoch.dt.date))
        
        df_orbits = df.apply(lambda x: self._select_orbit_in_year(row = x, yr = yr), axis = 1) 

        df_orbits.loc[df_orbits.InitOrbitId.notna(),'SemiMajorAxis'] =  \
                    df_orbits[df_orbits.InitOrbitId.notna()].apply(lambda x:
                        df_io[df_io.OrbitId == x['InitOrbitId']].SemiMajorAxis.values[0], axis = 1) 
        df_orbits.loc[df_orbits.InitOrbitId.notna(),'Inclination'] =  \
                    df_orbits[df_orbits.InitOrbitId.notna()].apply(lambda x:
                        df_io[df_io.OrbitId == x['InitOrbitId']].Inclination.values[0], axis = 1) 

        df_orbits.loc[df_orbits.DestOrbitId.notna(),'SemiMajorAxis'] =  \
                    df_orbits[df_orbits.DestOrbitId.notna()].apply(lambda x:
                        df_do[df_do.OrbitId == x['DestOrbitId']].SemiMajorAxis.values[0], axis = 1) 
        df_orbits.loc[df_orbits.DestOrbitId.notna(),'Inclination'] =  \
                    df_orbits[df_orbits.DestOrbitId.notna()].apply(lambda x:
                        df_do[df_do.OrbitId == x['DestOrbitId']].Inclination.values[0], axis = 1) 

        df_orbits = df_orbits[df_orbits.SemiMajorAxis.notna() | df_orbits.Inclination.notna()]
        
        for obj in df.ObjectType.unique():
            df_orbits.loc[df_orbits.DiscosId.isin(df[df.ObjectType == obj].DiscosId), 'ObjectType'] = obj
        
        

        sns.jointplot(data=xx, x="SemiMajorAxis", y="Inclination", kind="hex", xscale = 'log', bins = 'log', gridsize = 50, marginal_ticks = True, marginal_kws = {'bins' : 50, 'log_scale' : False}) 

        axes = xx[xx.SemiMajorAxis.notna()][['SemiMajorAxis','Inclination']].hist()
        axes[0,1].set_yscale('log')

        return df_orbits

    def select_pop_in_year(self, yr = None):
        """
        For a given year, extract only the objects in orbit during that year
        from the main objects dataframe (self.df)

        Args: 
            yr: The year for which satellite population data will be extracted
        Returns:
            df_yr_tot: A subset of the self.df dataframe for the given year
        """
        df = self.df
        df_launches = self.df_launches
        df_reentries = self.df_reentries
        df_frags = self.df_frags
        junk_obj = self.junk_obj
        junk_obj_str = '|'.join(junk_obj)

        # Assume current/most recent year if no year is given
        if yr == None:
            yr = df_launches.Epoch.dt.year.max()

        launchids_tot = df_launches[df_launches.Epoch.dt.year <= yr]
        launchids_tot = launchids_tot.LaunchId.unique()
        # Don't drop objects that reenter during the selected year.
        reentryids_tot = df_reentries[df_reentries.Epoch.dt.year < yr]
        reentryids_tot = reentryids_tot.ReentryId.unique()
        fragids_tot = df_frags[df_frags.Epoch.dt.year <= yr]
        fragids_tot = fragids_tot.FragmentationId.unique()

                    # Has been launched and not yet decayed, and
        df_yr_tot = df[(df.LaunchId.isin(launchids_tot) &
                    ~df.ReentryId.isin(reentryids_tot)) &
                    # (is either not a fragmentation object, or
                    # fragmentation event has already occurred)
                    (df.FragmentationId.isna() |
                    df.FragmentationId.isin(fragids_tot))].copy()
                        

        return df_yr_tot


    def assign_fragmentationid(self, df):
        """
        For a given dataframe with an OrbitId column, add a column
        'FragmentationId' for any fragmentation events involving each object, if
        any exist.

        Args: 
            df: A dataframe containing at least ObjectId and IntlDes columns.
        Returns:
            df: The given dataframe with an additional FragmentionId column,
            containing the Id of the most recent (if any) fragmention event
            associated with each object in the dataframe.
        """
        df_frags = self.df_frags
        df['FragmentationId'] = np.nan

        for i,row in df_frags.iterrows():
            frag_id = row.FragmentationId
            for ii in pd.eval(row.DiscosIds):
                intldes = df[df.DiscosID == int(ii)].IntlDes
                assert len(intldes) == 1
                intldes = intldes.iloc[0]
                if pd.isna(intldes):
                    continue
                intldes = re.sub(r'[A-Z]+', '', intldes)
                # Are there any objects with multiple fragmentation events?
                #assert pd.isna(df[df.IntlDes.notna() & df.IntlDes.str.contains(intldes)].FragmentationId).all() | (df[df.IntlDes.notna() & df.IntlDes.str.contains(intldes)].FragmentationId == frag_id).all()
                # For objects with multiple fragmentations, use the most recent
                if pd.notna(df[df.IntlDes.notna() & df.IntlDes.str.contains(intldes)].FragmentationId).all():
                    existing_id = df[df.IntlDes.notna() & df.IntlDes.str.contains(intldes)].FragmentationId
                    existing_max = df_frags[df_frags.FragmentationId.isin(existing_id)].Epoch.dt.date.max()
                    new_max = df_frags[df_frags.FragmentationId == frag_id].Epoch.dt.date.max()
                    # If current frag event is more recent to existing then leave
                    # it. Otherwise, assign the new frag Id
                    if new_max > existing_max:
                        continue
                    else:
                        df.loc[df.IntlDes.notna() & df.IntlDes.str.contains(intldes), 'FragmentationId'] = frag_id
                        
                else:
                    df.loc[df.IntlDes.notna() & df.IntlDes.str.contains(intldes), 'FragmentationId'] = frag_id
                

        return df



    def _select_orbit_in_year(self, row, yr):
        """
        For a row/series containing multiple Orbit ID's and epoch's for a single
        object, extract the most recent orbit relative to a given year. E.g. if
        row contains two Orbits with epoch 1991 and 1995 and yr = 1994, the
        Orbit Id and epoch with be returned for Orbit with epoch 1991.

        Args:
            row: row containing DiscosId, InitOrbitId, InitOrbitEpoch,
            DestOrbitId, DestOrbitEpoch.
            yr: the desired year to extract orbital information

        Returns:
            result: Series containing DiscosId, InitOrbitId, DestOrbitId,
            InitOrbitEpoch, and DestOrbitEpoch. There will be a single
            InitOrbitId or DestOrbitId and corresponding Init/DestOrbitEpoch,
            and the other values will be null, unless no orbital information
            exists for the given year, in which case all values other than
            DiscosId will be null.
        """

        result = pd.Series({  
                    'DiscosId'    : int(row['DiscosID']), 
                    'InitOrbitId' : np.nan, 
                    'DestOrbitId' : np.nan,
                    'InitOrbitEpoch' : np.nan,
                    'DestOrbitEpoch' : np.nan})

        if (row['InitOrbitId'] is np.nan) and (row['DestOrbitId'] is np.nan):
            return result

        if row['InitOrbitEpoch'] is not np.nan:
            row['InitOrbitEpoch'] = row['InitOrbitEpoch'][row['InitOrbitEpoch'] < datetime.date(yr+1,1,1)]
            if len(row['InitOrbitEpoch']) > 0: 
                idx = row['InitOrbitEpoch'].argmax()
                result['InitOrbitId'] = row['InitOrbitId'][idx]
                result['InitOrbitEpoch'] = row['InitOrbitEpoch'][idx]
        if row['DestOrbitEpoch'] is not np.nan:
            row['DestOrbitEpoch'] = row['DestOrbitEpoch'][row['DestOrbitEpoch'] < datetime.date(yr+1,1,1)]
            if len(row['DestOrbitEpoch']) > 0: 
                idx = row['DestOrbitEpoch'].argmax()
                result['DestOrbitId'] = row['DestOrbitId'][idx]
                result['DestOrbitEpoch'] = row['DestOrbitEpoch'][idx]

        epochs = [result['InitOrbitEpoch'], result['DestOrbitEpoch']]
        keys = [['InitOrbitId','InitOrbitEpoch'],['DestOrbitId','DestOrbitEpoch']]
        if np.sum([pd.notna(i) for i in epochs]) > 1:
            idx = np.array(epochs).argmax()
            result[keys[idx]] = np.nan

        return result

