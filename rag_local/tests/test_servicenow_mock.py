import responses
from agentic.servicenow_tool import ServiceNowTool

@responses.activate
def test_create_incident():
    responses.add(
        responses.POST,
        "https://dev200046.service-now.com/api/now/table/incident",
        json={"result": {"sys_id": "abc123", "number": "INC0001"}},
        status=201
    )
    sn = ServiceNowTool()
    res = sn.create_incident("short", "desc")
    assert res["sys_id"] == "abc123"
