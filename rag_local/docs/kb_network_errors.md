Title: Network Connectivity Problems
Tags: network, DNS, ERR_TIMEOUT

**Symptoms**
- ERR_TIMEOUT in logs
- Cannot reach API endpoints

**Troubleshooting**
1. Ping the host.
2. Verify DNS:
   `nslookup api.backend.local`
3. Check firewall rules.
4. Restart agent network module:
   `systemctl restart network-agent.service`
