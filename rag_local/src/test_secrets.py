# src/test_secrets.py
from agentic.secrets import get_servicenow_creds, get_n8n_webhook_url

def main():
    print("Testing secrets.py ...")
    try:
        inst, user, pw = get_servicenow_creds()
        print("ServiceNow instance:", inst)
        print("ServiceNow user    :", user)
        print("ServiceNow pw len  :", len(pw))
    except Exception as e:
        print("ServiceNow error   :", repr(e))

    n8n = get_n8n_webhook_url()
    print("N8N_WEBHOOK_URL    :", n8n or "(empty)")

if __name__ == "__main__":
    main()
