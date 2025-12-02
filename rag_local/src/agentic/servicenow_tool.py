# src/agentic/servicenow_tool.py
from typing import Optional, Dict, Any
import requests
from .secrets import get_servicenow_creds

SN_INSTANCE, SN_USER, SN_PASS = get_servicenow_creds()
BASE = f"https://{SN_INSTANCE}/api/now/table"


class ServiceNowTool:
    def __init__(self, timeout: int = 30):
        self.base = BASE
        self.auth = (SN_USER, SN_PASS)
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        self.timeout = timeout

    def create_incident(
        self,
        short_description: str,
        description: str = "",
        assignment_group_sysid: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create an incident in ServiceNow.
        """
        url = f"{self.base}/incident"
        payload: Dict[str, Any] = {
            "short_description": short_description,
            "description": description,
        }

        # Also show the description to the user as "Additional comments"
        if description:
            payload["comments"] = description

        if assignment_group_sysid:
            payload["assignment_group"] = assignment_group_sysid

        r = requests.post(
            url,
            auth=self.auth,
            headers=self.headers,
            json=payload,
            timeout=self.timeout,
        )
        # If SN returns error, this will raise with HTTP status
        r.raise_for_status()
        return r.json()["result"]

    def update_incident(self, sys_id: str, fields: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base}/incident/{sys_id}"
        r = requests.patch(
            url,
            auth=self.auth,
            headers=self.headers,
            json=fields,
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()["result"]

    def add_comment(self, sys_id: str, comment: str, work_note: bool = False) -> Dict[str, Any]:
        field = "work_notes" if work_note else "comments"
        return self.update_incident(sys_id, {field: comment})

    def find_group_sysid(self, group_name: str) -> Optional[str]:
        url = f"{self.base}/sys_user_group"
        params = {
            "sysparm_query": f"name={group_name}",
            "sysparm_limit": 1,
        }
        r = requests.get(
            url,
            auth=self.auth,
            headers=self.headers,
            params=params,
            timeout=self.timeout,
        )
        r.raise_for_status()
        res = r.json().get("result", [])
        if res:
            return res[0]["sys_id"]
        return None

    def get_incident_by_number(self, number: str) -> Optional[Dict[str, Any]]:
        url = f"{self.base}/incident"
        params = {
            "sysparm_query": f"number={number}",
            "sysparm_limit": 1,
        }
        r = requests.get(
            url,
            auth=self.auth,
            headers=self.headers,
            params=params,
            timeout=self.timeout,
        )
        r.raise_for_status()
        res = r.json().get("result", [])
        return res[0] if res else None
