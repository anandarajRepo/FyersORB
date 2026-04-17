# Browser-Based Authentication Guide

## Overview

The FyersORB trading strategy now includes **automatic browser-based authentication** that eliminates the need to manually copy-paste authorization codes. When you run `python main.py auth`, the system will:

1. **Auto-open your browser** with the Fyers login page
2. **Prompt for authentication** (Phone Number → OTP → PIN)
3. **Automatically capture the authorization code** 
4. **Generate and save the access token** to `.env`

## Quick Start

### For New Users (Initial Setup)

```bash
python main.py auth
```

This will:
1. Ask for your Fyers API credentials (Client ID and Secret Key)
2. Present authentication method options
3. Open a browser window automatically
4. Guide you through the login process

### For Existing Users (Re-authentication)

```bash
python main.py auth
```

If credentials already exist:
1. You'll be asked if you want to proceed with re-authentication
2. Choose authentication method (Browser-based or Manual)
3. Follow the browser-based flow

## How It Works

### Browser-Based Authentication Flow (Recommended)

```
1. Local Server Started
   └─ Listens on http://localhost:5000

2. Browser Opens Automatically
   └─ Fyers login page loads
   
3. Authentication in Browser
   ├─ Enter Phone Number
   ├─ Enter OTP (received on phone)
   └─ Enter Trading PIN
   
4. Automatic Redirect Capture
   └─ Local server receives auth code
   
5. Token Exchange
   └─ Auth code → Access Token
   
6. Automatic Save
   └─ Token saved to .env file
```

### Manual Authentication (Fallback)

If browser automation fails:

```
1. Browser opens (or manual URL provided)
2. Complete login and 2FA in browser
3. Copy authorization code from redirect URL
4. Paste code back in terminal
```

## Features

✅ **Automatic Browser Opening**
- Uses Python's `webbrowser` module
- Opens default system browser
- Works on Windows, macOS, Linux

✅ **Automatic Code Capture**
- Local HTTP server captures redirect
- No manual copy-paste needed
- Graceful timeout handling (5 minutes)

✅ **SEBI Compliant**
- Daily 2FA required
- No refresh tokens stored
- Secure PIN handling

✅ **User-Friendly**
- Clear on-screen instructions
- Success/error messages
- Profile information display

## Configuration

The system uses these environment variables:

```
FYERS_CLIENT_ID=your_client_id
FYERS_SECRET_KEY=your_secret_key
FYERS_REDIRECT_URI=http://localhost:5000/callback
FYERS_ACCESS_TOKEN=generated_token
FYERS_PIN=your_trading_pin
FYERS_LAST_AUTH_DATE=YYYY-MM-DD
```

## Troubleshooting

### Browser doesn't open automatically

**Solution:** The system will provide a URL to open manually. Copy and paste it into your browser.

### Port 5000 already in use

**Solution:** The code will automatically fall back to manual authentication mode. You can also change the port in the code:

```python
auth_manager.setup_browser_authentication(port=5001)  # Use different port
```

### Timeout waiting for authorization code

**Solution:** 
1. Make sure you completed the login in the browser
2. Check if the browser redirect happened correctly
3. Try again with a fresh authentication session

### Token validation failed

**Solution:**
1. Verify your trading PIN is correct
2. Check if your Fyers account has 2FA enabled
3. Ensure your IP is registered (if required by Fyers)

## Authentication Methods Comparison

| Feature | Browser-Based | Manual Entry |
|---------|---------------|--------------|
| Browser Opens | Automatic | Manual |
| Code Capture | Automatic | Copy-Paste |
| Speed | 1-2 minutes | 2-3 minutes |
| Error Handling | Built-in | Manual |
| Suitable for Scripts | Yes | Limited |

## Security Notes

⚠️ **Important Security Practices:**

1. **PIN is saved locally** - Only enter PIN if your machine is secure
2. **Access tokens expire** - Daily re-authentication required per SEBI rules
3. **Don't share credentials** - Client ID and Secret Key are sensitive
4. **Use HTTPS URLs** - Ensure redirect URIs are secure in production

## Example Output

```
================================================================================
FYERS API DAILY 2FA AUTHENTICATION SETUP (SEBI-COMPLIANT)
================================================================================
Found existing API credentials in environment
This will perform FULL re-authentication (not refresh)

Proceed with full re-authentication? (y/n) [y]: y

Authentication Method:
1. Browser-based (recommended) - Browser opens automatically
2. Manual code entry - Copy/paste authorization code

Select authentication method (1/2) [default: 1]: 1

Starting browser-based authentication flow...

================================================================================
FYERS API DAILY 2FA AUTHENTICATION (BROWSER-BASED)
================================================================================
A browser window will open automatically for authentication.
Refresh tokens are disabled per SEBI regulations (effective April 1, 2026).

Starting local authorization server on port 5000...
Opening browser for authentication...
Auth URL: https://api-t1.fyers.in/api/v3/generate-authcode?...

Waiting for authentication...
1. Enter your phone number when prompted
2. Verify the OTP sent to your phone
3. Enter your trading PIN
4. The authorization code will be captured automatically

✓ Authorization code received!
Exchanging authorization code for access token...
✓ Access token obtained successfully!

Saving access token...
✓ Access token saved to .env file

AUTHENTICATION SUCCESSFUL!
✓ Access token is valid and ready to use
Account: John Doe
Email: john@example.com
```

## Running the Strategy After Authentication

Once authenticated:

```bash
# Test WebSocket connection
python main.py test

# Run the trading strategy
python main.py run
```

## API Reference

### FyersAuthManager Methods

```python
# Browser-based authentication
auth_manager.setup_browser_authentication(port=5000)

# Manual authentication (fallback)
auth_manager.setup_full_authentication()

# Check token validity
auth_manager.is_token_valid(access_token)

# Get valid token (handles SEBI daily 2FA)
auth_manager.get_valid_access_token()
```

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify your Fyers API credentials
3. Ensure your phone has internet for OTP delivery
4. Check network connectivity for local redirect server

## What's Next

After successful authentication:
- The access token is valid for trading
- Daily re-authentication required per SEBI rules
- Token automatically refreshed when needed
- Start running the ORB strategy with `python main.py run`
