# Fyers Authentication Methods Guide

## Overview

The FyersORB trading strategy now includes **three authentication methods**:

1. **CLI-Only (No Browser)** - Complete authentication in terminal ⭐ NEW
2. **Browser-Based** - Automatic browser opening with redirect capture
3. **Manual Code Entry** - Browser login with manual code copy-paste

Choose the method that works best for your environment!

## Quick Start

### For New Users (Initial Setup)

```bash
python main.py auth
```

You'll be presented with three authentication options:

```
======================================================================
AUTHENTICATION METHOD SELECTION
======================================================================

1. CLI-Only (No Browser) - Complete in terminal
   - Provide phone, password, OTP, PIN via command line
   - Paste authorization code from browser redirect

2. Browser-Based (Recommended) - Browser opens automatically
   - Browser opens with login page
   - Authorization code captured automatically

3. Manual Code Entry - Copy/paste from browser
   - Open URL manually in browser
   - Copy authorization code and paste here

Select authentication method (1/2/3) [default: 1]:
```

---

## Method 1: CLI-Only Authentication (No Browser) ⭐ NEW

### Overview
Complete all authentication steps in the command line terminal. No browser opening required. Perfect for:
- Automated scripts and CI/CD pipelines
- Remote server environments
- Headless systems
- Users preferring terminal-only interaction

### Flow

```
STEP 1: LOGIN INFORMATION
├─ Display authentication URL (for reference)

STEP 2: ENTER YOUR CREDENTIALS
├─ Phone Number / Email
└─ Password (hidden input)

STEP 3: VERIFY OTP
├─ OTP sent to registered phone
└─ Enter 6-digit OTP

STEP 4: ENTER TRADING PIN
├─ Enter PIN (4-10 digits, hidden)
├─ Confirm PIN
└─ PIN saved to .env

STEP 5: GET AUTHORIZATION CODE
├─ Option 1: Manual browser login
│            Get code from redirect URL
│
└─ Paste authorization code

STEP 6: EXCHANGE CODE FOR TOKEN
└─ Authorization code exchanged for access token

STEP 7: SAVE TOKEN
└─ Access token saved to .env
```

### Example Session

```bash
$ python main.py auth

================================================================================
FYERS API DAILY 2FA AUTHENTICATION SETUP (SEBI-COMPLIANT)
================================================================================

AUTHENTICATION METHOD SELECTION
=====================================================================
1. CLI-Only (No Browser) - Complete in terminal
2. Browser-Based (Recommended) - Browser opens automatically
3. Manual Code Entry - Copy/paste from browser

Select authentication method (1/2/3) [default: 1]: 1

Starting CLI-only authentication flow...

FYERS API DAILY 2FA AUTHENTICATION (CLI-ONLY)
===============================================================
Complete authentication entirely in the terminal.

STEP 1: LOGIN INFORMATION
===============================================================
If needed, the authentication URL is:
https://api-t1.fyers.in/api/v3/generate-authcode?client_id=...

STEP 2: ENTER YOUR CREDENTIALS
===============================================================
Enter your Fyers login credentials:
 Phone Number / Email: 9876543210
 Password: 

STEP 3: VERIFY OTP
===============================================================
An OTP has been sent to your registered phone number.
Check your SMS and enter the OTP below.
 Enter OTP (6 digits): 123456

STEP 4: ENTER TRADING PIN
===============================================================
Enter your Fyers trading PIN (4-6 digits).
 Trading PIN: 
 Confirm Trading PIN: 
 
 ✓ Trading PIN saved to .env file

STEP 5: GET AUTHORIZATION CODE
===============================================================
You have two options to proceed:

1. MANUAL OPTION:
   - Visit the authentication URL in your browser
   - Login with the credentials you provided above
   - Complete the 2FA process
   - Copy the authorization code from redirect URL
   - Paste it below

2. DIRECT API OPTION (if supported):
   - Attempt direct API authentication

Select option (1/2) [default: 1]: 1
 
 After completing authentication in your browser:
 Paste the authorization code here: eyJ0eXAiOiJKV1QiLCJhbGc...

STEP 6: EXCHANGE CODE FOR TOKEN
===============================================================
 Exchanging authorization code for access token...
 ✓ Access token obtained successfully!

STEP 7: SAVE TOKEN
===============================================================
 ✓ Access token saved to .env file

AUTHENTICATION SUCCESSFUL!
===============================================================
 ✓ Access token is valid and ready to use
 Account Name: John Doe
 Email: john@example.com
 User ID: USER123456

 Ready to run strategy: python main.py run
```

### Advantages

✅ **No Browser Required** - Run on any system (servers, headless, SSH)  
✅ **Scriptable** - Can be automated in pipelines  
✅ **Terminal Only** - No dependency on display servers  
✅ **Full Control** - See every step in terminal output  
✅ **Same Security** - 2FA verification still required  

### Use Cases

- Automated trading bot deployment
- Cloud server environments (AWS, Azure, GCP)
- CI/CD pipeline authentication
- Docker containers
- Remote SSH sessions
- Headless Linux systems

---

## Method 2: Browser-Based Authentication

### Overview
Automatic browser opening with automatic redirect code capture. Most seamless for desktop users.

### Features

- 🌐 **Auto-Opens Browser** - Uses system default browser
- 🔄 **Auto-Captures Code** - Local redirect server captures auth code
- ⏱️ **Fast** - 1-2 minutes total
- 🎯 **User-Friendly** - Clear visual feedback

### Flow

```
Local Server Started (port 5000)
        ↓
Browser Opens Automatically
        ↓
Complete Login in Browser:
├─ Enter Phone Number
├─ Verify OTP (SMS)
└─ Enter Trading PIN
        ↓
Browser Redirects with auth code
        ↓
Auth Code Captured Automatically
        ↓
Access Token Generated
        ↓
Token Saved to .env
```

### Advantages

✅ **Automatic Code Capture** - No copy-paste  
✅ **Fast** - Quickest method  
✅ **Visual Feedback** - Browser success page  
✅ **Recommended for Desktop** - Best user experience  

### Requirements

- Default browser must be available
- Port 5000 must be free
- Network connectivity

---

## Method 3: Manual Code Entry

### Overview
Open browser manually, copy authorization code from redirect URL, paste in terminal.

### Flow

```
Display Auth URL
        ↓
Manual Browser Open
        ↓
Complete Login & 2FA
        ↓
Extract Code from Redirect URL
        ↓
Paste Code in Terminal
        ↓
Token Generation & Save
```

### Advantages

✅ **Reliable** - Works anywhere  
✅ **Flexible** - Works with different redirect URIs  
✅ **Portable** - No port forwarding needed  

---

## Authentication Methods Comparison

| Feature | CLI-Only | Browser-Based | Manual |
|---------|----------|---------------|--------|
| **Browser Required** | No | Yes (auto) | Yes (manual) |
| **Code Capture** | Manual paste | Automatic | Manual paste |
| **Speed** | 2-3 min | 1-2 min | 2-3 min |
| **Scriptable** | ✅ Yes | ❌ Limited | ❌ No |
| **Desktop UX** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Server Deploy** | ⭐⭐⭐⭐⭐ | ❌ No | ⭐⭐⭐ |
| **Port Needed** | None | 5000 | None |

---

## Troubleshooting

### CLI-Only Issues

**Problem:** Password input not working
- **Solution:** Try using visible input mode if secure input fails

**Problem:** OTP rejected  
- **Solution:** OTP expires in 10 minutes, request a new one

**Problem:** Authorization code won't exchange
- **Solution:** Code expires quickly, paste immediately after receiving it

### Browser-Based Issues

**Problem:** Browser doesn't open
- **Solution:** URL provided in output, open manually
- **Fallback:** Use Manual Code Entry method

**Problem:** Port 5000 in use
- **Solution:** Close other applications on that port

**Problem:** Timeout waiting for code
- **Solution:** Complete authentication in browser within 5 minutes

### General Issues

**Problem:** Token validation failed
- **Solution:** Check internet connection, try again

**Problem:** Invalid authorization code
- **Solution:** Code expires quickly, try authentication again

**Problem:** Wrong PIN error
- **Solution:** Verify PIN carefully

---

## Security Best Practices

🔒 **Important:**

1. **PIN Storage** - PIN saved locally to .env only on secure machines
2. **Terminal History** - CLI input may appear in shell history
3. **Credentials** - Never share Client ID or Secret Key
4. **Token Expiry** - Tokens expire per SEBI rules, re-authenticate daily
5. **HTTPS** - Ensure URLs use HTTPS

---

## Environment Variables

After successful authentication:

```bash
FYERS_CLIENT_ID=your_client_id          # From Fyers dashboard
FYERS_SECRET_KEY=your_secret_key        # From Fyers dashboard
FYERS_ACCESS_TOKEN=generated_token      # Automatically saved
FYERS_PIN=your_trading_pin              # Saved if using pin option
FYERS_LAST_AUTH_DATE=YYYY-MM-DD        # Tracks daily 2FA requirement
```

---

## Running the Strategy After Authentication

Once authenticated:

```bash
# Test WebSocket connection
python main.py test

# Run the trading strategy
python main.py run

# Re-authenticate (daily required per SEBI)
python main.py auth
```

---

## API Reference

### FyersAuthManager Methods

```python
# CLI-Only authentication (NEW)
auth_manager.setup_cli_authentication()

# Browser-based authentication
auth_manager.setup_browser_authentication(port=5000)

# Manual authentication
auth_manager.setup_full_authentication()

# Check token validity
auth_manager.is_token_valid(access_token)

# Get valid token (handles SEBI daily 2FA)
auth_manager.get_valid_access_token()
```

---

## Recommendations

**For Desktop Users:**
→ Use Method 2: Browser-Based (fastest, most seamless)

**For Server/Cloud Deployment:**
→ Use Method 1: CLI-Only ⭐ (no browser dependency)

**For Troubleshooting:**
→ Use Method 3: Manual Code Entry (most reliable)

**For Automation/Scripts:**
→ Use Method 1: CLI-Only ⭐ (scriptable, repeatable)

**For CI/CD Pipelines:**
→ Use Method 1: CLI-Only ⭐ (can be automated with inputs)

---

## What's Next

After successful authentication:
- Access token is valid for trading
- Daily re-authentication required per SEBI rules
- Start running the ORB strategy: `python main.py run`
- Monitor trading performance and adjust parameters as needed
