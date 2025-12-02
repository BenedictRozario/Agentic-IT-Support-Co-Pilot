Title: Login Troubleshooting
Tags: login, authentication, user

**Symptoms**
- Unable to log in
- Invalid credentials error
- Login screen loops

**Troubleshooting Steps**
1. Verify username and password.
2. Ensure user is active and not locked.
3. Reset password and retry.
4. If ERR_AUTH_FAIL in logs:
   - Restart auth service: `systemctl restart auth.service`
   - Verify downstream token if API authentication.
