import importlib

import pandas as pd
import pytest

fetch_module = importlib.import_module(
    "pyroads.segmenter._util.fetch_road_network_info"
)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeSession:
    calls = []
    responses = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def mount(self, prefix, adapter):
        return None

    def get(self, url, params, timeout):
        self.calls.append((url, params.copy()))
        return FakeResponse(self.responses.pop(0))


def test_fetch_road_network_info_paginates_with_stable_order(monkeypatch):
    FakeSession.calls = []
    FakeSession.responses = [
        {"count": 2},
        {
            "features": [{"attributes": {"ROAD": "H001", "OBJECTID": 1}}],
            "exceededTransferLimit": True,
        },
        {"features": [{"attributes": {"ROAD": "H002", "OBJECTID": 2}}]},
    ]
    monkeypatch.setattr(fetch_module.requests, "Session", FakeSession)

    query_params = {"where": "ROAD LIKE 'H%'"}
    result = fetch_module.fetch_road_network_info(
        query_params=query_params,
        additional_params={"returnGeometry": False},
        outFields="ROAD,OBJECTID",
    )

    assert isinstance(result, pd.DataFrame)
    assert result["ROAD"].tolist() == ["H001", "H002"]
    assert query_params == {"where": "ROAD LIKE 'H%'"}
    assert FakeSession.calls[0][1]["returnCountOnly"] is True
    assert FakeSession.calls[1][1]["orderByFields"] == "OBJECTID"
    assert FakeSession.calls[1][1]["resultOffset"] == 0
    assert FakeSession.calls[2][1]["resultOffset"] == 1


def test_fetch_road_network_info_uses_current_main_roads_endpoint(monkeypatch):
    FakeSession.calls = []
    FakeSession.responses = [{"count": 0}, {"features": []}]
    monkeypatch.setattr(fetch_module.requests, "Session", FakeSession)

    fetch_module.fetch_road_network_info()

    assert FakeSession.calls[0][0] == (
        "https://gisservices.mainroads.wa.gov.au/arcgis/rest/services/"
        "OpenData/RoadAssets_DataPortal/MapServer/17/query"
    )


def test_fetch_road_network_info_rejects_invalid_chunk_limit():
    with pytest.raises(ValueError, match="greater than zero"):
        fetch_module.fetch_road_network_info(chunk_limit=0)


def test_fetch_road_network_info_rejects_invalid_request_timeout():
    with pytest.raises(ValueError, match="request_timeout must be greater than zero"):
        fetch_module.fetch_road_network_info(request_timeout=0)


def test_fetch_road_network_info_rejects_empty_continuation_page(monkeypatch):
    FakeSession.responses = [{"count": 2}, {"features": [], "exceededTransferLimit": True}]
    monkeypatch.setattr(fetch_module.requests, "Session", FakeSession)

    with pytest.raises(RuntimeError, match="empty page"):
        fetch_module.fetch_road_network_info()
