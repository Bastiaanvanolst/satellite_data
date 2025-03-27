"""
Will fetch and store DISCOS data locally. Can fetch from the following databases:
 - objects
 - launches
 - reentries
 - launch-sites
 - launch-systems
 - launch-vehicles
 - initial-orbits
 - destination-orbits
 - fragmentations
 - fragmentation-event-types
 - entities
 - propellants

Usage example;
import sat_data
d = sat_data.get_data(database = 'objects')
"""

import datetime as dt
import time
import glob
import re
import os
import io
import requests
import numpy as np
import pandas as pd
from dotenv import load_dotenv


def load_esa_token():
    """Load ESA token from .env file or environment variable"""
    load_dotenv()  # Load environment variables from .env file
    token_file = ".esa_token.txt"
    if os.path.exists(token_file):
        with open(token_file, "r") as f:
            return f.read().strip()
    return os.getenv("ESA_TOKEN")


# Initialize ESA token
esa_token = load_esa_token()
if not esa_token:
    raise ValueError(
        "ESA token not found. Please provide it in .esa_token.txt or set ESA_TOKEN environment variable"
    )

# Create esa_data directory if it doesn't exist
os.makedirs("./esa_data", exist_ok=True)


def get_data(database):
    """
    Check whether data exists in file and is recent.
    If so, read in. If not, retrieve from API and save.
    Return DataFrame
    """
    max_age_days = 365
    filelist = glob.glob(os.path.join("./esa_data", f"{database}_*"))

    # If no previous files exist, retrieve and write a new one
    if len(filelist) == 0:
        df = retrieve_discos_data(database)
        write_data(df=df, prefix=database)
        return df

    # Check if existing files are recent. If so, read in most recent, otherwise, retrieve and write new one.
    min_delta_t = dt.timedelta(days=max_age_days)
    recent_file = None
    for f in filelist:
        f_date = re.findall(r"\d{4}-\d{2}-\d{2}", f)
        if len(f_date) > 0:
            f_date = f_date[-1]
        else:
            continue
        f_date = dt.datetime.strptime(f_date, "%Y-%m-%d")
        delta_t = dt.datetime.now() - f_date
        if delta_t < min_delta_t:
            min_delta_t = delta_t
            recent_file = f

    if recent_file is None:
        df = retrieve_discos_data(database)
        write_data(df=df, prefix=database)
    else:
        df = read_data(recent_file)

    return df


def write_data(df, prefix):
    """
    Write the data to file with appropriate date and prefix
    """

    datestr = dt.datetime.now().strftime("%Y-%m-%d")
    output_file = f"./esa_data/{prefix}_{datestr}.csv"

    # Write the data
    df.to_csv(output_file, index=False)

    print(f"Data written to file {output_file}")

    return


def read_data(filename):
    """
    Read data from csv file
    """

    df = pd.read_csv(filename)

    if "Epoch" in df.columns:
        df["Epoch"] = pd.to_datetime(df["Epoch"])

    print(f"Read data from file {filename}")

    return df


def retrieve_discos_data(database):
    """
    Retrieve data from DISCOS API with proper rate limiting and error handling
    """
    discos_url = "https://discosweb.esoc.esa.int"
    db_url = f"{discos_url}/api/{database}"

    print("No recent files found, attempting to retrieve data from")
    print(discos_url)
    print("...")

    discos_headers = {"Authorization": f"Bearer {esa_token}"}
    params = discos_params(database)

    def make_request():
        """Helper function to make API request with rate limit handling"""
        response = requests.get(
            db_url,
            headers=discos_headers,
            params=params,
            timeout=30,  # 30 second timeout
        )

        if not response.ok:
            error_msg = response.json().get("error", "Unknown error")
            raise ConnectionError(f"Discos request failed: {error_msg}")

        # Handle rate limiting
        try:
            limit_remain = int(response.headers.get("X-Ratelimit-Remaining", 0))
            if limit_remain == 0:
                wait_time = float(response.headers.get("Retry-After", 60)) + 5
                print(f"Rate limit reached. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
                return make_request()  # Retry after waiting
        except (ValueError, TypeError) as e:
            print(f"Warning: Could not parse rate limit headers: {e}")
            # Continue with the response even if we can't parse rate limit info

        return response

    # Get first page
    response = make_request()
    dat = response.json()["data"]
    df = pd.json_normalize(dat)
    last_page = response.json()["meta"]["pagination"]["totalPages"]

    # Get remaining pages
    dfs = [df]
    for page in range(2, last_page + 1):
        print(f"{page} / {last_page}")
        params["page[number]"] = page
        response = make_request()
        dat = response.json()["data"]
        dfs.append(pd.json_normalize(dat))

    # Concatenate all DataFrames at once
    df = pd.concat(dfs, ignore_index=True)
    df = clean_discos(database=database, df=df)
    print("Data retrieved successfully")

    return df


def clean_discos(database, df):
    clean = {
        "objects": clean_discos_objects,
        "launches": clean_discos_launches,
        "reentries": clean_discos_reentries,
        "launch-sites": clean_discos_launchsites,
        "launch-systems": clean_discos_launchsystems,
        "launch-vehicles": clean_discos_launchvehicles,
        "initial-orbits": clean_discos_initialorbits,
        "destination-orbits": clean_discos_destinationorbits,
        "fragmentations": clean_discos_fragmentations,
        "fragmentation-event-types": clean_discos_fragmentationeventtypes,
        "entities": clean_discos_entities,
        "propellants": clean_discos_propellants,
    }

    return clean[database](df)


def clean_discos_objects(df):
    drop_cols = [
        "type",
        "relationships.states.links.self",
        "relationships.states.links.related",
        "relationships.initialOrbits.links.self",
        "relationships.initialOrbits.links.related",
        "relationships.launch.links.self",
        "relationships.launch.links.related",
        "relationships.launch.data.type",
        "relationships.reentry.links.self",
        "relationships.reentry.links.related",
        "relationships.reentry.data.type",
        "relationships.operators.links.self",
        "relationships.operators.links.related",
        "relationships.destinationOrbits.links.self",
        "relationships.destinationOrbits.links.related",
        "relationships.reentry.data",
        "relationships.launch.data",
        "links.self",
    ]

    rename_cols = {
        "id": "DiscosID",
        "attributes.cosparId": "IntlDes",
        "attributes.xSectAvg": "XSectAvg",
        "attributes.depth": "Depth",
        "attributes.xSectMin": "XSectMin",
        "attributes.vimpelId": "VimpelId",
        "attributes.shape": "Shape",
        "attributes.satno": "SatNo",
        "attributes.name": "SatName",
        "attributes.height": "Height",
        "attributes.objectClass": "ObjectType",
        "attributes.mass": "Mass",
        "attributes.xSectMax": "XSectMax",
        "attributes.length": "Length",
        "relationships.initialOrbits.data": "InitOrbitId",
        "relationships.launch.data.id": "LaunchId",
        "relationships.reentry.data.id": "ReentryId",
        "relationships.operators.data": "OperatorId",
        "relationships.destinationOrbits.data": "DestOrbitId",
    }

    df.drop(columns=drop_cols, inplace=True)
    df.rename(columns=rename_cols, inplace=True)

    def safe_id_convert(x):
        if pd.isna(x) or x is None:
            return np.nan
        if isinstance(x, list):
            return [int(xi["id"]) for xi in x]
        if isinstance(x, dict):
            return int(x["id"])
        return np.nan

    df["InitOrbitId"] = df["InitOrbitId"].apply(safe_id_convert)
    df["DestOrbitId"] = df["DestOrbitId"].apply(safe_id_convert)
    df["OperatorId"] = df["OperatorId"].apply(safe_id_convert)
    df["VimpelId"] = df["VimpelId"].apply(
        lambda x: np.nan if pd.isna(x) or x is None else int(x)
    )

    return df


def clean_discos_launches(df):
    drop_cols = [
        "type",
        "relationships.site.links.self",
        "relationships.site.links.related",
        "relationships.site.data.type",
        "relationships.objects.links.self",
        "relationships.objects.links.related",
        "relationships.entities.links.self",
        "relationships.entities.links.related",
        "relationships.vehicle.links.self",
        "relationships.vehicle.links.related",
        "relationships.vehicle.data.type",
        "relationships.site.data",
        "relationships.vehicle.data",
        "links.self",
    ]

    rename_cols = {
        "id": "LaunchId",
        "relationships.site.data.id": "LaunchSiteId",
        "attributes.epoch": "Epoch",
        "attributes.flightNo": "FlightNo",
        "attributes.failure": "Failure",
        "attributes.cosparLaunchNo": "CosparLaunchNo",
        "relationships.objects.data": "ObjectId",
        "relationships.vehicle.data.id": "VehicleId",
        "relationships.entities.data": "Entities",
    }

    df.drop(columns=drop_cols, inplace=True)
    df.rename(columns=rename_cols, inplace=True)

    df["LaunchSiteId"] = df["LaunchSiteId"].apply(
        lambda x: np.nan if x is None else str(x)
    )
    df["CountryId"] = df["Entities"].apply(
        lambda x: (
            np.nan
            if not x
            else (
                np.nan
                if True not in [i["type"] == "country" for i in x]
                else np.array([i["id"] for i in x])[[i["type"] == "country" for i in x]]
            )
        )
    )
    df["OrganisationId"] = df["Entities"].apply(
        lambda x: (
            np.nan
            if not x
            else (
                np.nan
                if True not in [i["type"] == "organisation" for i in x]
                else np.array([i["id"] for i in x])[
                    [i["type"] == "organisation" for i in x]
                ]
            )
        )
    )
    df.drop(columns="Entities", inplace=True)
    df["ObjectId"] = df["ObjectId"].apply(
        lambda x: np.nan if not x else [xi["id"] for xi in x]
    )

    df["Epoch"] = pd.to_datetime(df["Epoch"])

    return df


def clean_discos_launchsystems(df):
    drop_cols = [
        "type",
        "relationships.entities.links.self",
        "relationships.entities.links.related",
        "relationships.families.links.self",
        "relationships.families.links.related",
        "links.self",
    ]

    rename_cols = {
        "id": "LaunchSystemId",
        "relationships.entities.data": "Entities",
        "relationships.families.data": "VehicleFamilyId",
        "attributes.name": "VehicleFamilyName",
    }

    df.drop(columns=drop_cols, inplace=True)
    df.rename(columns=rename_cols, inplace=True)

    df["LaunchSystemId"] = df["LaunchSystemId"].apply(
        lambda x: np.nan if x is None else str(x)
    )
    df["CountryId"] = df["Entities"].apply(
        lambda x: (
            np.nan
            if not x
            else (
                np.nan
                if True not in [i["type"] == "country" for i in x]
                else np.array([i["id"] for i in x])[[i["type"] == "country" for i in x]]
            )
        )
    )
    df["OrganisationId"] = df["Entities"].apply(
        lambda x: (
            np.nan
            if not x
            else (
                np.nan
                if True not in [i["type"] == "organisation" for i in x]
                else np.array([i["id"] for i in x])[
                    [i["type"] == "organisation" for i in x]
                ]
            )
        )
    )
    df.drop(columns="Entities", inplace=True)
    df["VehicleFamilyId"] = df["VehicleFamilyId"].apply(
        lambda x: np.nan if not x else [xi["id"] for xi in x]
    )

    return df


def clean_discos_launchvehicles(df):
    drop_cols = [
        "type",
        "relationships.launches.links.self",
        "relationships.launches.links.related",
        "relationships.engines.links.self",
        "relationships.engines.links.related",
        "relationships.family.links.self",
        "relationships.family.links.related",
        "relationships.family.data.type",
        "relationships.stages.links.self",
        "relationships.stages.links.related",
        "links.self",
    ]

    rename_cols = {
        "id": "VehicleId",
        "attributes.leoCapacity": "LEOCapacity",
        "attributes.geoCapacity": "GEOCapacity",
        "attributes.name": "VehicleName",
        "attributes.numStages": "NumStages",
        "attributes.gtoCapacity": "GTOCapacity",
        "attributes.escCapacity": "ESCCapacity",
        "attributes.successfulLaunches": "SuccessfulLaunches",
        "attributes.failedLaunches": "FailedLaunches",
        "attributes.ssoCapacity": "SSOCapacity",
        "attributes.mass": "Mass",
        "attributes.height": "Height",
        "attributes.thrustLevel": "ThrustLevel",
        "attributes.diameter": "Diameter",
        "relationships.launches.data": "LaunchId",
        "relationships.engines.data": "EngineId",
        "relationships.family.data.id": "FamilyId",
        "relationships.stages.data": "StageId",
    }

    df.drop(columns=drop_cols, inplace=True)
    df.rename(columns=rename_cols, inplace=True)

    df["VehicleId"] = df["VehicleId"].apply(lambda x: np.nan if x is None else str(x))
    df["LaunchId"] = df["LaunchId"].apply(
        lambda x: np.nan if not x else [xi["id"] for xi in x]
    )
    df["EngineId"] = df["EngineId"].apply(
        lambda x: np.nan if not x else [xi["id"] for xi in x]
    )
    df["StageId"] = df["StageId"].apply(
        lambda x: np.nan if not x else [xi["id"] for xi in x]
    )

    return df


def clean_discos_reentries(df):
    drop_cols = [
        "type",
        "relationships.objects.links.self",
        "relationships.objects.links.related",
        "links.self",
    ]

    rename_cols = {"id": "ReentryId", "attributes.epoch": "Epoch"}

    df.drop(columns=drop_cols, inplace=True)
    df.rename(columns=rename_cols, inplace=True)
    df["Epoch"] = pd.to_datetime(df["Epoch"])

    return df


def clean_discos_launchsites(df):
    drop_cols = [
        "type",
        "relationships.launches.links.self",
        "relationships.launches.links.related",
        "relationships.operators.links.self",
        "relationships.operators.links.related",
        "links.self",
    ]

    rename_cols = {
        "id": "LaunchSiteId",
        "attributes.constraints": "Constraints",
        "attributes.pads": "Pads",
        "attributes.altitude": "Altitude",
        "attributes.latitude": "Latitude",
        "attributes.azimuths": "Azimuths",
        "attributes.name": "Name",
        "attributes.longitude": "Longitude",
    }

    df.drop(columns=drop_cols, inplace=True)
    df.rename(columns=rename_cols, inplace=True)

    return df


def clean_discos_initialorbits(df):
    drop_cols = [
        "type",
        "relationships.object.links.self",
        "relationships.object.links.related",
        "links.self",
    ]

    rename_cols = {
        "id": "OrbitId",
        "attributes.sma": "SemiMajorAxis",
        "attributes.epoch": "Epoch",
        "attributes.aPer": "ArgPeriapsis",
        "attributes.inc": "Inclination",
        "attributes.mAno": "MeanAnomoly",
        "attributes.ecc": "Eccentricity",
        "attributes.raan": "RAAN",
        "attributes.frame": "RefFrame",
    }

    df.drop(columns=drop_cols, inplace=True)
    df.rename(columns=rename_cols, inplace=True)

    return df


def clean_discos_destinationorbits(df):
    drop_cols = [
        "type",
        "relationships.object.links.self",
        "relationships.object.links.related",
        "links.self",
    ]

    rename_cols = {
        "id": "OrbitId",
        "attributes.sma": "SemiMajorAxis",
        "attributes.epoch": "Epoch",
        "attributes.aPer": "ArgPeriapsis",
        "attributes.inc": "Inclination",
        "attributes.mAno": "MeanAnomoly",
        "attributes.ecc": "Eccentricity",
        "attributes.raan": "RAAN",
        "attributes.frame": "RefFrame",
    }

    df.drop(columns=drop_cols, inplace=True)
    df.rename(columns=rename_cols, inplace=True)

    return df


def clean_discos_fragmentations(df):

    drop_cols = [
        "type",
        "relationships.objects.links.self",
        "relationships.objects.links.related",
        "links.self",
    ]

    rename_cols = {
        "id": "FragmentationId",
        "attributes.eventType": "eventType",
        "attributes.longitude": "Longitude",
        "attributes.comment": "Comment",
        "attributes.epoch": "Epoch",
        "attributes.latitude": "Latitude",
        "attributes.altitude": "Altitude",
        "relationships.objects.data": "DiscosIds",
    }

    df.drop(columns=drop_cols, inplace=True)
    df.rename(columns=rename_cols, inplace=True)

    df["Epoch"] = pd.to_datetime(df["Epoch"])
    df["DiscosIds"] = df["DiscosIds"].apply(lambda x: list(int(xx["id"]) for xx in x))

    return df


def clean_discos_entities(df):
    drop_cols = [
        "type",
        "relationships.objects.links.self",
        "relationships.objects.links.related",
        "relationships.launchSites.links.self",
        "relationships.launchSites.links.related",
        "relationships.hostCountry.links.self",
        "relationships.hostCountry.links.related",
        "relationships.launches.links.self",
        "relationships.launches.links.related",
        "relationships.launchSystems.links.self",
        "relationships.launchSystems.links.related",
        "attributes.dateRange",
        "links.self",
    ]
    rename_cols = {
        "id": "EntityId",
        "relationships.objects.data": "ObjectIds",
        "relationships.launchSites.data": "LaunchSites",
        # fetching hostCountry relationship from API seems not functional (dec 2020)
        #'relationships.hostCountry.data',
        "relationships.launches.data": "LaunchIds",
        "relationships.launchSystems.data": "LaunchSystems",
        "attributes.name": "Name",
        "attributes.dateRange.empty": "DateEmpty",
        "attributes.dateRange.upper": "DateUpper",
        "attributes.dateRange.lowerInc": "DateLowerInc",
        "attributes.dateRange.upperInc": "DateUpperInc",
        "attributes.dateRange.lower": "DateLower",
        "attributes.dateRange.display": "DateRange",
        #'attributes.dateRange',
    }

    df.drop(columns=drop_cols, inplace=True)
    df.rename(columns=rename_cols, inplace=True)
    df["DateUpper"] = pd.to_datetime(df["DateUpper"])
    # Entity 400: Alma Mater Studiorum Universita di Bologna was formed in 1088,
    # which is below the minimum (outside the range that can be represented by
    # nanosecond-resolution). For now, errors = 'ignore' will leave this date as
    # a str
    df["DateLower"] = pd.to_datetime(df["DateLower"], errors="ignore")
    df["ObjectIds"] = df["ObjectIds"].apply(
        lambda x: np.nan if not x else [xi["id"] for xi in x]
    )

    return df


def clean_discos_propellants(df):
    drop_cols = [
        "type",
        "relationships.stages.links.self",
        "relationships.stages.links.related",
        "links.self",
    ]
    rename_cols = {
        "id": "PropellantId",
        "attributes.oxidiser": "Oxidiser",
        "attributes.fuel": "Fuel",
        "attributes.solidPropellant": "SolidPropellant",
        "relationships.stages.data": "StageId",
    }

    df.drop(columns=drop_cols, inplace=True)
    df.rename(columns=rename_cols, inplace=True)
    df["StageId"] = df["StageId"].apply(
        lambda x: np.nan if not x else [xi["id"] for xi in x]
    )

    return df


def clean_discos_fragmentationeventtypes(df):
    drop_cols = [
        "type",
        "links.self",
    ]
    rename_cols = {
        "id": "FragEventId",
        "attributes.name": "ObjectName",
    }

    df.drop(columns=drop_cols, inplace=True)
    df.rename(columns=rename_cols, inplace=True)

    return df


def discos_params(database):
    """Get API parameters for different DISCOS databases"""
    params = {
        "page[number]": 1,
        "page[size]": 100,
    }

    if database == "objects":
        params.update(
            {
                "include": "launch,reentry,initialOrbits,destinationOrbits,operators",
                "sort": "satno",
            }
        )
    elif database == "launches":
        params.update(
            {
                "include": "site,vehicle,objects,entities",
            }
        )
    elif database == "launch-systems":
        params.update(
            {
                "include": "families,entities",
            }
        )
    elif database == "launch-sites":
        pass  # No additional parameters needed
    elif database == "initial-orbits":
        pass  # No additional parameters needed
    elif database == "destination-orbits":
        pass  # No additional parameters needed
    elif database == "fragmentation-event-types":
        pass  # No additional parameters needed
    elif database == "fragmentations":
        params.update(
            {
                "include": "objects",
            }
        )
    elif database == "reentries":
        pass  # No additional parameters needed
    elif database == "entities":
        params.update(
            {
                "include": "objects,launchSites,launches,launchSystems",
            }
        )
    elif database == "launch-vehicles":
        params.update(
            {
                "include": "launches,engines,family,stages",
            }
        )
    elif database == "propellants":
        params.update(
            {
                "include": "stages",
            }
        )
    else:
        raise ValueError(f"Unknown database: {database}")

    return params


def get_ucsdata():

    now = dt.datetime.now()
    filelist = glob.glob(os.path.join("./esa_data", "ucsdata_*"))

    if len(filelist) != 0:
        dates = [re.findall(r"\d{4}-\d{2}-\d{2}", f)[0] for f in filelist]
        dates = [dt.datetime.strptime(d, "%Y-%m-%d") for d in dates]
        delta_t = [now - d for d in dates]
        recent_i = int(np.argmin(delta_t))
        recent_file = filelist[recent_i]
        recent_date = dates[recent_i].strftime("%Y-%m-%d")

        print("UCS data file found, generated on {}".format(recent_date))
        print("File: {}".format(recent_file))
        print("")
        df = pd.read_csv(recent_file)
    else:
        print("No saved UCS data files found, generating new data ...")

        with requests.Session() as session:
            # run the session in a with block to force session to close if we exit

            # need to log in first. note that we get a 200 to say the web site got the data, not that we are logged in

            resp = session.get("https://www.ucsusa.org/media/11492")
            if resp.status_code != 200:
                print(resp)
                raise ConnectionError(f"GET fail on request for Box Score:\n{resp}")

            df = pd.read_excel(io.BytesIO(resp.content))

            print("Data retrieved")

            col_rename = {
                "Name of Satellite, Alternate Names": "SatName",
                "Country of Operator/Owner": "Country",
                "Country/Org of UN Registry": "UNRegCountry",
                "Operator/Owner": "Owner",
                "Users": "Users",
                "Purpose": "Purpose",
                "Detailed Purpose": "PurposeDetailed",
                "Class of Orbit": "OrbitClass",
                "Type of Orbit": "OrbitType",
                "Longitude of GEO (degrees)": "LongitudeGEO",
                "Perigee (km)": "Perigee",
                "Apogee (km)": "Apogee",
                "Eccentricity": "Eccentricity",
                "Inclination (degrees)": "Inclination",
                "Period (minutes)": "Period",
                "Launch Mass (kg.)": "MassLaunch",
                "Dry Mass (kg.)": "MassDry",
                "Power (watts)": "Power",
                "Date of Launch": "Launch",
                "Expected Lifetime (yrs.)": "ExpLifetime",
                "Contractor": "Contractor",
                "Country of Contractor": "ContractorCountry",
                "Launch Site": "LaunchSite",
                "Launch Vehicle": "LaunchVehicle",
                "COSPAR Number": "IntlDes",
                "NORAD Number": "NORAD",
                "Comments": "Comments",
            }

            df.rename(columns=col_rename, inplace=True)

            df.to_csv(
                "./esa_data/ucsdata_" + now.strftime("%Y-%m-%d") + ".csv", index=False
            )

            print("Data saved to file.")

            session.close()

    df["Launch"] = pd.to_datetime(df.Launch)
    df["LaunchYear"] = df.Launch.dt.year

    return df
