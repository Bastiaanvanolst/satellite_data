from pprint import pprint
import requests

esa_token = (
    "IjIzZGM3ZmFlLWFhNTktNDFkOC1hZjVkLTZmMTAwZmVkZWQxNSI.vZPDne1Z-KHiE2lMX7fVRU5Yhq4"
)
esa_url = "https://discosweb.esoc.esa.int"


response = requests.get(
    f"{esa_url}/api/objects",
    headers={
        "Authorization": f"Bearer {esa_token}",
        "DiscosWeb-Api-Version": "2",
    },
)

doc = response.json()
if response.ok:
    pprint(doc["data"])
else:
    pprint(doc["errors"])
