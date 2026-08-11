from typing import Any, Dict, Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


DATA_SOURCE_URL = "https://gisservices.mainroads.wa.gov.au/arcgis/rest/services/OpenData/RoadAssets_DataPortal/MapServer/17/query"

DEFAULT_PARAMETERS = {
    "where": "1=1",
    "outFields": ",".join([
        "ROAD",
        "START_SLK",
        "END_SLK",
        "CWY",
        "NETWORK_TYPE",
        "START_TRUE_DIST",
        "END_TRUE_DIST",
        "RA_NO",
    ]),
    "outSR": 4326,
    "f": "json",
    "orderByFields": "OBJECTID",
    "returnGeometry": False,
}

def fetch_road_network_info(
    url: str = DATA_SOURCE_URL,
    chunk_limit: Optional[int] = None,
    query_params: Optional[Dict[str, Any]] = None,
    additional_params: Optional[Dict[str, Any]] = None,
    request_timeout: float = 30.0,
    **kwargs: Any,
) -> pd.DataFrame:
    """Download the Main Roads WA road network as a DataFrame.

    ``query_params`` replaces the defaults, while ``additional_params`` and
    legacy keyword arguments override individual default parameters. Results
    are fetched in ArcGIS pages ordered by ``OBJECTID`` so records cannot be
    skipped or duplicated when the service pages a changing dataset.
    """
    if chunk_limit is not None and chunk_limit <= 0:
        raise ValueError("chunk_limit must be greater than zero")
    if request_timeout <= 0:
        raise ValueError("request_timeout must be greater than zero")

    params = dict(DEFAULT_PARAMETERS if query_params is None else query_params)
    if additional_params:
        params.update(additional_params)
    params.update(kwargs)
    params.setdefault("orderByFields", "OBJECTID")

    retry = Retry(
        total=3,
        connect=3,
        read=3,
        status=3,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )

    def request_json(session: requests.Session, request_params: Dict[str, Any]) -> Dict[str, Any]:
        response = session.get(url, params=request_params, timeout=request_timeout)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("ArcGIS response was not a JSON object")
        if "error" in payload:
            raise ValueError(f"ArcGIS request failed: {payload['error']}")
        return payload

    with requests.Session() as session:
        session.mount("https://", HTTPAdapter(max_retries=retry))
        session.mount("http://", HTTPAdapter(max_retries=retry))
        count_params = {**params, "returnCountOnly": True}
        count_payload = request_json(session, count_params)
        if "count" not in count_payload:
            raise ValueError(f"ArcGIS response did not contain a record count: {count_payload}")
        record_count = count_payload["count"]

        print(
            f"Downloading {record_count} records"
            + (":" if chunk_limit is None else f", chunk_limit={chunk_limit}:")
        )

        features = []
        offset = 0
        chunk_counter = 0
        last_payload: Dict[str, Any] = {"features": []}

        while chunk_limit is None or chunk_counter < chunk_limit:
            chunk_counter += 1
            last_payload = request_json(
                session,
                {**params, "resultOffset": offset},
            )
            page_features = last_payload.get("features")
            if not isinstance(page_features, list):
                raise ValueError(f"ArcGIS response did not contain a feature list: {last_payload}")
            if not page_features:
                if last_payload.get("exceededTransferLimit"):
                    raise RuntimeError("ArcGIS returned an empty page while more records were expected")
                break

            features.extend(page_features)
            offset += len(page_features)
            if not last_payload.get("exceededTransferLimit", False):
                break

    print(f"\nDownload Completed. received {len(features)} records")
    last_payload["features"] = features
    result = pd.json_normalize(last_payload, record_path="features")
    result.columns = [c.replace("attributes.", "") for c in result.columns]

    return result