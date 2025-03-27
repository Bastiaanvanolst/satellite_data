import sat_data
import os
import time
from dotenv import load_dotenv
import requests


def test_discos_data():
    print("Testing DISCOS data retrieval...")

    # Load and verify token
    load_dotenv()
    token = os.getenv("ESA_TOKEN")
    if token:
        print("ESA token found in environment variables")
    else:
        print("No ESA token found in environment variables")
        return

    # Try to fetch a small subset of objects data
    try:
        print("\nFetching objects data...")
        # Store original params function and requests.get
        original_params = sat_data.discos_params
        original_get = requests.get

        # Create new params function that uses the maximum allowed page size
        def new_params(database):
            params = original_params(database)
            params["page[size]"] = 100  # Maximum allowed page size
            return params

        sat_data.discos_params = new_params
        print("Making API request to DISCOS...")

        # Test the API endpoint directly first
        discos_url = "https://discosweb.esoc.esa.int"
        db_url = f"{discos_url}/api/objects"
        headers = {"Authorization": f"Bearer {token}"}
        params = new_params("objects")

        print(f"Testing direct API request to: {db_url}")
        response = requests.get(
            db_url, headers=headers, params=params, timeout=60
        )  # Increased timeout
        print(f"Response status code: {response.status_code}")
        if response.ok:
            print("Direct API request successful!")
            print("Total pages:", response.json()["meta"]["pagination"]["totalPages"])
        else:
            print(f"Response content: {response.text}")
            return

        # Modify the sat_data module's request behavior
        def make_request_with_retry(*args, **kwargs):
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    if "timeout" not in kwargs:
                        kwargs["timeout"] = 60
                    response = original_get(*args, **kwargs)
                    if response.ok:
                        time.sleep(0.5)  # Add a small delay between requests
                        return response
                    else:
                        print(f"Request failed with status {response.status_code}")
                        if attempt < max_retries - 1:
                            time.sleep(2)  # Wait before retrying
                            continue
                        raise ConnectionError(
                            f"Request failed after {max_retries} attempts"
                        )
                except requests.Timeout:
                    print(f"Request timed out (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                    raise
            return None

        # Replace the requests.get in sat_data with our modified version
        requests.get = make_request_with_retry

        objects_df = sat_data.get_data(database="objects")
        print(f"Successfully retrieved {len(objects_df)} objects")
        print("\nSample of objects data:")
        print(objects_df[["SatName", "ObjectType", "Mass"]].head())
    except Exception as e:
        print(f"Error fetching objects data: {e}")
        print("Full error details:", str(e))
    finally:
        # Restore original params function and requests.get
        sat_data.discos_params = original_params
        requests.get = original_get

    # Try to fetch a small subset of launches data
    try:
        print("\nFetching launches data...")
        print("Making API request to DISCOS...")
        launches_df = sat_data.get_data(database="launches")
        print(f"Successfully retrieved {len(launches_df)} launches")
        print("\nSample of launches data:")
        print(launches_df[["Epoch", "FlightNo", "Failure"]].head())
    except Exception as e:
        print(f"Error fetching launches data: {e}")
        print("Full error details:", str(e))


if __name__ == "__main__":
    test_discos_data()
