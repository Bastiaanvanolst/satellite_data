"""
Plotting and analysis of the DISCOS data retrieved by sat_data.py

Usage example;
import sat_plots
p = sat_plots.SatPlots()
p......
"""

import datetime
import re
import ast
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
import numpy as np

import sat_data

# TODO understand missing values in datasets. Payloads without launch id? Government or commercial? Increasing occurrence?
# TODO add handling of undetermined orbit class
# TODO consider decay_year - launch_year as a measurement of lifetime.


class SatPlots(object):
    def __init__(self):
        self.G = 6.67430e-11  # Gravitational constant (m,kg,s)
        self.M_earth = 5.972e24  # Earth's mass (kg)
        self.R_earth = 6371000  # Earth's radius (m)

        self.set_plot_options()

        self.df_frags = sat_data.get_data(database="fragmentations")
        self.df_fragevents = sat_data.get_data(database="fragmentation-event-types")
        df = sat_data.get_data(database="objects")
        # convert values stored as list (read_csv imports them as string)
        df["InitOrbitId"] = df.InitOrbitId.apply(
            lambda x: np.nan if x is np.nan else list(ast.literal_eval(x))
        )
        df["DestOrbitId"] = df.DestOrbitId.apply(
            lambda x: np.nan if x is np.nan else list(ast.literal_eval(x))
        )
        df = self.assign_fragmentationid(df)
        self.df = df
        self.df_launches = sat_data.get_data(database="launches")
        self.df_reentries = sat_data.get_data(database="reentries")
        self.df_launchsites = sat_data.get_data(database="launch-sites")
        self.df_launchsystems = sat_data.get_data(database="launch-systems")
        self.df_launchvehicles = sat_data.get_data(database="launch-vehicles")
        self.df_propellants = sat_data.get_data(database="propellants")
        self.df_entities = sat_data.get_data(database="entities")
        self.df_initorbits = sat_data.get_data(database="initial-orbits")
        # self.df_initorbits = self.assign_orbit_class(df_initorbits)
        self.df_destorbits = sat_data.get_data(database="destination-orbits")
        # self.df_destorbits = self.assign_orbit_class(df_destorbits)
        self.df_ucs = sat_data.get_ucsdata()

        self.junk_obj = [
            "Rocket Body",
            "Unknown",
            "Rocket Mission Related Object",
            "Payload Mission Related Object",
            "Rocket Fragmentation Debris",
            "Payload Fragmentation Debris",
            "Payload Debris",
            "Rocket Debris",
            "Other Debris",
            "Other Mission Related Object",
        ]

    def set_plot_options(self):
        rcParams["figure.titlesize"] = 12
        rcParams["axes.labelsize"] = 10
        rcParams["xtick.labelsize"] = 10
        rcParams["ytick.labelsize"] = 10
        rcParams["ytick.major.size"] = 0.0
        rcParams["legend.fontsize"] = 10
        rcParams["axes.grid.axis"] = "y"
        rcParams["axes.grid"] = True
        rcParams["grid.color"] = "xkcd:grey"
        rcParams["grid.linestyle"] = "-"
        rcParams["grid.linewidth"] = 0.5
        rcParams["grid.alpha"] = 0.4

        self.plot_colors = [
            "lightsteelblue",
            "slategrey",
            "lightcoral",
            "black",
            "gold",
            "crimson",
            "dodgerblue",
        ]

    def plot_pop_evolution(
        self,
        vars=["Payload", "Launches", "Junk", "Reentries", "Fragmentations"],
        type="net",
        scale="symlog",
    ):

        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(7, 5))
        fig.set_tight_layout({"rect": (0, 0, 1, 0.95)})
        fig.suptitle(f"Evolution of the Orbital Population; total ({type})")

        dfs = self.calc_growth_per_year()
        type_dict = {"change": int(0), "net": int(1)}
        df = dfs[type_dict[type]]
        plot_colors = self.plot_colors

        for i, var in enumerate(vars):
            ax.plot(
                df.Year,
                df[var],
                color=plot_colors[i],
                marker="o",
                markersize=3,
                markeredgecolor=None,
                clip_on=False,
                label=var,
            )

        ax.set_ylabel(f"Total ({type}) per year")
        ax.legend()
        ax.set_yscale(scale)
        if scale != "log":
            ax.set_ylim(bottom=0.0)
        self._hide_spines([ax])
        self._add_data_source(
            ax=ax,
            text="European Space Agency\n(https://discosweb.esoc.esa.int)",
            x=0.95,
            y=0.02,
            ha="right",
        )

    def plot_orbit_density(self, yr=2020):
        """
        Plots the number density of satellites for a given year, as a function
        of SemiMajorAxis and Inclination
        """
        df_io = self.df_initorbits
        df_do = self.df_destorbits
        df = self.select_pop_in_year(yr=yr)

        df["InitOrbitEpoch"] = df.InitOrbitId.apply(
            lambda x: (
                np.nan
                if x is np.nan
                else np.array(df_io[df_io.OrbitId.isin(x)].Epoch.dt.date)
            )
        )
        df["DestOrbitEpoch"] = df.DestOrbitId.apply(
            lambda x: (
                np.nan
                if x is np.nan
                else np.array(df_do[df_do.OrbitId.isin(x)].Epoch.dt.date)
            )
        )

        df_orbits = df.apply(lambda x: self._select_orbit_in_year(row=x, yr=yr), axis=1)

        df_orbits.loc[df_orbits.InitOrbitId.notna(), "SemiMajorAxis"] = df_orbits[
            df_orbits.InitOrbitId.notna()
        ].apply(
            lambda x: df_io[df_io.OrbitId == x["InitOrbitId"]].SemiMajorAxis.values[0],
            axis=1,
        )
        df_orbits.loc[df_orbits.InitOrbitId.notna(), "Inclination"] = df_orbits[
            df_orbits.InitOrbitId.notna()
        ].apply(
            lambda x: df_io[df_io.OrbitId == x["InitOrbitId"]].Inclination.values[0],
            axis=1,
        )

        df_orbits.loc[df_orbits.DestOrbitId.notna(), "SemiMajorAxis"] = df_orbits[
            df_orbits.DestOrbitId.notna()
        ].apply(
            lambda x: df_do[df_do.OrbitId == x["DestOrbitId"]].SemiMajorAxis.values[0],
            axis=1,
        )
        df_orbits.loc[df_orbits.DestOrbitId.notna(), "Inclination"] = df_orbits[
            df_orbits.DestOrbitId.notna()
        ].apply(
            lambda x: df_do[df_do.OrbitId == x["DestOrbitId"]].Inclination.values[0],
            axis=1,
        )

        df_orbits = df_orbits[
            df_orbits.SemiMajorAxis.notna() | df_orbits.Inclination.notna()
        ]

        for obj in df.ObjectType.unique():
            df_orbits.loc[
                df_orbits.DiscosId.isin(df[df.ObjectType == obj].DiscosId), "ObjectType"
            ] = obj

        # sns.jointplot(data=xx, x="SemiMajorAxis", y="Inclination", kind="hex", xscale = 'log', bins = 'log', gridsize = 50, marginal_ticks = True, marginal_kws = {'bins' : 50, 'log_scale' : False})

        # axes = xx[xx.SemiMajorAxis.notna()][['SemiMajorAxis','Inclination']].hist()
        # axes[0,1].set_yscale('log')

        return df_orbits

    def plot_purpose_ucs(self):
        df_ucs = self.df_ucs

        fig = plt.figure(figsize=(10, 5))

        df_ucs.loc[df_ucs.Purpose == "Earth Observation ", "Purpose"] = (
            "Earth Observation"
        )
        df_ucs.loc[df_ucs.Purpose == "Earth/Space Observation", "Purpose"] = (
            "Earth & Space Observation"
        )

        main_purposes = [
            "Communications",
            "Earth Observation",
            "Technology Development",
            "Navigation/Global Positioning",
            "Space Science",
            "Earth Science",
            "Navigation/Regional Positioning",
            "Technology Demonstration",
        ]

        df_ucs["PurposeBinned"] = df_ucs.Purpose.apply(
            lambda x: (
                "Unknown"
                if pd.isna(x)
                else (
                    "Multipurpose"
                    if (x.split("/")[0] in main_purposes and len(x.split("/")) > 1)
                    else (x if x in main_purposes else "Other")
                )
            ),
        )

        counts = df_ucs.PurposeBinned.value_counts()

        ax = counts.plot.barh(zorder=4)
        ax.set_title("Purposes of active satellites by number")
        ax.grid(False, axis="y")
        ax.grid(True, axis="x", zorder=0)
        self._hide_spines([ax])
        # ax.tick_params(labelrotation = 30, axis = 'y')
        # ax.set_figure(fig)
        fig.set_tight_layout({"rect": (0, 0, 1, 0.95)})

        self._add_data_source(
            ax=ax,
            text="Union of Concerned Scientists,\nucsusa.org/resources/satellite-database",
        )

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
        df["FragmentationId"] = np.nan

        for i, row in df_frags.iterrows():
            frag_id = row.FragmentationId
            for ii in pd.eval(row.DiscosIds):
                intldes = df[df.DiscosID == int(ii)].IntlDes
                assert len(intldes) == 1
                intldes = intldes.iloc[0]
                if pd.isna(intldes):
                    continue
                intldes = re.sub(r"[A-Z]+", "", intldes)
                # Are there any objects with multiple fragmentation events?
                # assert pd.isna(df[df.IntlDes.notna() & df.IntlDes.str.contains(intldes)].FragmentationId).all() | (df[df.IntlDes.notna() & df.IntlDes.str.contains(intldes)].FragmentationId == frag_id).all()
                # For objects with multiple fragmentations, use the most recent
                if pd.notna(
                    df[
                        df.IntlDes.notna() & df.IntlDes.str.contains(intldes)
                    ].FragmentationId
                ).all():
                    existing_id = df[
                        df.IntlDes.notna() & df.IntlDes.str.contains(intldes)
                    ].FragmentationId
                    existing_max = df_frags[
                        df_frags.FragmentationId.isin(existing_id)
                    ].Epoch.dt.date.max()
                    new_max = df_frags[
                        df_frags.FragmentationId == frag_id
                    ].Epoch.dt.date.max()
                    # If current frag event is more recent to existing then leave
                    # it. Otherwise, assign the new frag Id
                    if new_max > existing_max:
                        continue
                    else:
                        df.loc[
                            df.IntlDes.notna() & df.IntlDes.str.contains(intldes),
                            "FragmentationId",
                        ] = frag_id

                else:
                    df.loc[
                        df.IntlDes.notna() & df.IntlDes.str.contains(intldes),
                        "FragmentationId",
                    ] = frag_id

        return df

    def calc_growth_per_year(self):
        """
        Calculate the population evolution for each year, and return a dataframe
        for the yearly change and net total in each year

        Returns:
            ...
        """

        df = self.df
        df_launches = self.df_launches
        df_reentries = self.df_reentries
        df_frags = self.df_frags
        junk_obj = self.junk_obj

        yr_min = int(np.nanmin(df_launches.Epoch.dt.year.unique()))
        yr_max = int(np.nanmax(df_launches.Epoch.dt.year.unique()))
        yrs = [i for i in range(yr_min, yr_max + 1)]

        obj_types = list(df.ObjectType.unique())
        new_cols = ["Year", "Launches", "Reentries", "Fragmentations"] + obj_types

        df_yr_change = pd.DataFrame(
            np.zeros((len(yrs), len(new_cols))), columns=new_cols
        )
        df_yr_change["Year"] = yrs
        df_yr_total = pd.DataFrame(
            np.zeros((len(yrs), len(new_cols))), columns=new_cols
        )
        df_yr_total["Year"] = yrs

        for yr in yrs:
            launchids = df_launches[df_launches.Epoch.dt.year == yr]
            launchids = launchids.LaunchId.unique()
            launchids_tot = df_launches[df_launches.Epoch.dt.year <= yr]
            reentryids = df_reentries[df_reentries.Epoch.dt.year == yr]
            reentryids = reentryids.ReentryId.unique()
            reentryids_tot = df_reentries[df_reentries.Epoch.dt.year <= yr]
            fragids = df_frags[df_frags.Epoch.dt.year == yr]
            fragids = fragids.FragmentationId.unique()
            fragids_tot = df_frags[df_frags.Epoch.dt.year <= yr]

            # Yearly change
            df_yr = self.select_launched_in_year(yr)
            counts = df_yr.groupby("ObjectType")["DiscosID"].count()
            counts = [counts[obj] if obj in counts.keys() else 0 for obj in obj_types]
            df_yr_change.loc[df_yr_change.Year == yr, obj_types] = counts
            df_yr_change.loc[df_yr_change.Year == yr, "Launches"] = len(launchids)
            df_yr_change.loc[df_yr_change.Year == yr, "Reentries"] = len(reentryids)
            df_yr_change.loc[df_yr_change.Year == yr, "Fragmentations"] = len(fragids)

            # Yearly totals (net)
            df_yr_tot = self.select_pop_in_year(yr)
            counts_tot = df_yr_tot.groupby("ObjectType")["DiscosID"].count()
            counts_tot = [
                counts_tot[obj] if obj in counts_tot.keys() else 0 for obj in obj_types
            ]
            df_yr_total.loc[df_yr_total.Year == yr, obj_types] = counts_tot
            df_yr_total.loc[df_yr_total.Year == yr, "Launches"] = len(launchids_tot)
            df_yr_total.loc[df_yr_total.Year == yr, "Reentries"] = len(reentryids_tot)
            df_yr_total.loc[df_yr_total.Year == yr, "Fragmentations"] = len(fragids_tot)

        df_yr_change["Junk"] = df_yr_change[junk_obj].sum(axis=1)
        df_yr_total["Junk"] = df_yr_total[junk_obj].sum(axis=1)

        return df_yr_change, df_yr_total

    def select_pop_in_year(self, yr=None):
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
        junk_obj_str = "|".join(junk_obj)

        # Assume current/most recent year if no year is given
        if yr is None:
            yr = df_launches.Epoch.dt.year.max()

        launchids_tot = df_launches[df_launches.Epoch.dt.year <= yr]
        launchids_tot = launchids_tot.LaunchId.unique()
        # Don't drop objects that reenter during the selected year.
        reentryids_tot = df_reentries[df_reentries.Epoch.dt.year < yr]
        reentryids_tot = reentryids_tot.ReentryId.unique()
        fragids_tot = df_frags[df_frags.Epoch.dt.year <= yr]
        fragids_tot = fragids_tot.FragmentationId.unique()

        # Has been launched and not yet decayed, and
        df_yr_tot = df[
            (
                df.LaunchId.isin(launchids_tot)
                & ~df.ReentryId.isin(reentryids_tot)
                &
                # (is either not a fragmentation object, or
                # fragmentation event has already occurred and not decayed)
                df.FragmentationId.isna()
            )
            | (
                df.FragmentationId.isin(fragids_tot)
                & ~df.ReentryId.isin(reentryids_tot)
            )
        ].copy()

        return df_yr_tot

    def select_launched_in_year(self, yr=None):
        """
        For a given year, extact only the objects launched in that year from the
        main objects dataframe
        """
        df = self.df
        df_launches = self.df_launches
        df_reentries = self.df_reentries
        df_frags = self.df_frags
        junk_obj = self.junk_obj
        junk_obj_str = "|".join(junk_obj)

        # Assume current/most recent year if no year is given
        if yr is None:
            yr = df_launches.Epoch.dt.year.max()

        launchids = df_launches[df_launches.Epoch.dt.year == yr]
        launchids = launchids.LaunchId.unique()
        fragids = df_frags[df_frags.Epoch.dt.year == yr]
        fragids = fragids.FragmentationId.unique()

        # Currently, fragmentation objects are assigned the same LaunchId as
        # their progenitor, even though they technically do not become a unique
        # object until the fragmentation event. So, we must take this into
        # consideration for extracting the correct objects in each year

        df_yr = df[
            (df.LaunchId.isin(launchids) & df.FragmentationId.isna())
            | df.FragmentationId.isin(fragids)
        ].copy()
        # This will not count Payloads/Rocket Bodies that are involved in
        # fragmentation events in a later year, but is preferable to incorrectly
        # counting fragmentation objects prior to the frag event. To improve,
        # need some way to reliably distinguish between fragmentation
        # progenitors and debris (IntlDes? ObjectType?)

        return df_yr

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

        result = pd.Series(
            {
                "DiscosId": int(row["DiscosID"]),
                "InitOrbitId": np.nan,
                "DestOrbitId": np.nan,
                "InitOrbitEpoch": np.nan,
                "DestOrbitEpoch": np.nan,
            }
        )

        if (row["InitOrbitId"] is np.nan) and (row["DestOrbitId"] is np.nan):
            return result

        if row["InitOrbitEpoch"] is not np.nan:
            row["InitOrbitEpoch"] = row["InitOrbitEpoch"][
                row["InitOrbitEpoch"] < datetime.date(yr + 1, 1, 1)
            ]
            if len(row["InitOrbitEpoch"]) > 0:
                idx = row["InitOrbitEpoch"].argmax()
                result["InitOrbitId"] = row["InitOrbitId"][idx]
                result["InitOrbitEpoch"] = row["InitOrbitEpoch"][idx]
        if row["DestOrbitEpoch"] is not np.nan:
            row["DestOrbitEpoch"] = row["DestOrbitEpoch"][
                row["DestOrbitEpoch"] < datetime.date(yr + 1, 1, 1)
            ]
            if len(row["DestOrbitEpoch"]) > 0:
                idx = row["DestOrbitEpoch"].argmax()
                result["DestOrbitId"] = row["DestOrbitId"][idx]
                result["DestOrbitEpoch"] = row["DestOrbitEpoch"][idx]

        epochs = [result["InitOrbitEpoch"], result["DestOrbitEpoch"]]
        keys = [["InitOrbitId", "InitOrbitEpoch"], ["DestOrbitId", "DestOrbitEpoch"]]
        if np.sum([pd.notna(i) for i in epochs]) > 1:
            idx = np.array(epochs).argmax()
            result[keys[idx]] = np.nan

        return result

    def _hide_spines(self, axlist=None):
        """
        Hides the top and rightmost axis spines from view for all active
        figures and their respective axes.
        """

        if axlist is None:
            return

        for ax in axlist:
            # Disable spines.
            ax.spines["right"].set_color("none")
            ax.spines["left"].set_color("none")
            ax.spines["top"].set_color("none")
            ax.spines["bottom"].set_color("xkcd:grey")
            # Disable ticks.
            ax.xaxis.set_ticks_position("bottom")

    def _add_data_source(self, ax=None, text=None, x=0.95, y=0.9, ha="right"):

        text = "Data source: " + text

        ax.text(
            x,
            y,
            text,
            transform=ax.transAxes,
            fontsize=7,
            alpha=0.6,
            ha=ha,
            zorder=1000,
        )
