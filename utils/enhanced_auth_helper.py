# utils/enhanced_auth_helper.py

"""
Enhanced Fyers Authentication Helper for ORB Trading Strategy
Provides comprehensive authentication with auto-refresh, PIN support, and error handling
Includes browser automation and automatic redirect URL capture
"""

import hashlib
import requests
import logging
import os
import json
import getpass
import sys
import time
import datetime
import webbrowser
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from typing import Optional, Tuple, Dict, Any

logger = logging.getLogger(__name__)


class AuthCodeRedirectHandler(BaseHTTPRequestHandler):
    """Handles the OAuth redirect and captures the authorization code"""

    auth_code = None
    redirect_received = threading.Event()

    def do_GET(self):
        """Handle the redirect callback"""
        parsed_url = urlparse(self.path)
        query_params = parse_qs(parsed_url.query)

        # Extract auth code from redirect
        if 'code' in query_params:
            AuthCodeRedirectHandler.auth_code = query_params['code'][0]
            AuthCodeRedirectHandler.redirect_received.set()

            # Send success response to browser
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            success_html = """
            <html>
            <head><title>Authentication Successful</title></head>
            <body style="font-family: Arial, sans-serif; text-align: center; padding: 50px;">
                <h1 style="color: green;">✓ Authentication Successful</h1>
                <p>Authorization code received. You can close this window and return to the terminal.</p>
                <p style="color: #666;">The trading strategy will now proceed with token generation.</p>
            </body>
            </html>
            """
            self.wfile.write(success_html.encode())
        else:
            # Auth code not found
            self.send_response(400)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            error_html = """
            <html>
            <head><title>Authentication Failed</title></head>
            <body style="font-family: Arial, sans-serif; text-align: center; padding: 50px;">
                <h1 style="color: red;">✗ Authentication Failed</h1>
                <p>No authorization code received. Please try again.</p>
            </body>
            </html>
            """
            self.wfile.write(error_html.encode())

    def log_message(self, format, *args):
        """Suppress default logging of HTTP requests"""
        pass


class FyersAuthManager:
    """Fyers authentication manager — SEBI-compliant daily 2FA (no refresh tokens)"""

    def __init__(self):
        self.client_id = os.environ.get('FYERS_CLIENT_ID')
        self.secret_key = os.environ.get('FYERS_SECRET_KEY')
        self.redirect_uri = os.environ.get('FYERS_REDIRECT_URI', "https://trade.fyers.in/api-login/redirect-to-app")
        self.access_token = os.environ.get('FYERS_ACCESS_TOKEN')
        self.pin = os.environ.get('FYERS_PIN')

        # API endpoints
        self.auth_url = "https://api-t1.fyers.in/api/v3/generate-authcode"
        self.token_url = "https://api-t1.fyers.in/api/v3/validate-authcode"
        self.profile_url = "https://api-t1.fyers.in/api/v3/profile"

    def save_to_env(self, key: str, value: str) -> bool:
        """Save or update environment variable in .env file"""
        try:
            # Use absolute path so the correct .env is found regardless of CWD
            # (e.g. when invoked from a cron job that cd's to a different directory)
            env_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')

            # Read existing .env file
            env_vars = {}
            if os.path.exists(env_file):
                with open(env_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and '=' in line and not line.startswith('#'):
                            k, v = line.split('=', 1)
                            env_vars[k] = v

            # Update the specific key
            env_vars[key] = value

            # Write back to .env file
            with open(env_file, 'w', encoding='utf-8') as f:
                for k, v in env_vars.items():
                    f.write(f"{k}={v}\n")

            # Update current environment
            os.environ[key] = value

            logger.debug(f"Successfully saved {key} to .env file")
            return True

        except Exception as e:
            logger.error(f"Error saving {key} to .env file: {e}")
            return False

    def _secure_input(self, prompt: str, max_attempts: int = 3) -> str:
        """Get secure input with fallback to regular input"""
        for attempt in range(max_attempts):
            try:
                # Try getpass first (more secure)
                value = getpass.getpass(prompt).strip()
                if value:
                    return value
                else:
                    print("Input cannot be empty. Please try again.")

            except (EOFError, KeyboardInterrupt):
                print("\nInput cancelled by user")
                raise
            except Exception as e:
                if attempt < max_attempts - 1:
                    print(f"Secure input failed ({e}), trying regular input...")
                    try:
                        value = input(prompt.replace(":", " (visible): ")).strip()
                        if value:
                            return value
                        else:
                            print("Input cannot be empty. Please try again.")
                    except (EOFError, KeyboardInterrupt):
                        print("\nInput cancelled by user")
                        raise
                else:
                    print(f"All input methods failed: {e}")
                    raise ValueError("Could not get secure input")

        raise ValueError("Maximum attempts exceeded")

    def get_or_request_pin(self) -> str:
        """Get PIN from environment or request from user with better input handling"""
        if self.pin and len(self.pin) >= 4:
            logger.debug("Using PIN from environment")
            return self.pin

        print("\n" + "=" * 60)
        print("TRADING PIN REQUIRED")
        print("=" * 60)
        print("Your Fyers trading PIN is required for secure authentication.")
        print("This PIN will be saved securely in your .env file for future use.")
        print("The PIN is needed for automatic token refresh functionality.")

        max_attempts = 3
        for attempt in range(max_attempts):
            print(f"\nAttempt {attempt + 1}/{max_attempts}")

            try:
                pin = self._secure_input("Enter your Fyers trading PIN: ")

                # Basic validation
                if not pin.isdigit():
                    print(" PIN must contain only numbers")
                    continue

                if len(pin) < 4:
                    print(" PIN must be at least 4 digits")
                    continue

                if len(pin) > 10:
                    print(" PIN seems too long (max 10 digits)")
                    continue

                # Confirm PIN
                confirm_pin = self._secure_input("Confirm your trading PIN: ")

                if pin != confirm_pin:
                    print(" PINs do not match! Please try again.")
                    continue

                # Save PIN to environment for future use
                if self.save_to_env('FYERS_PIN', pin):
                    self.pin = pin
                    print(" PIN saved successfully to .env file")
                    return pin
                else:
                    print("️  PIN validation successful but couldn't save to .env file")
                    return pin

            except (EOFError, KeyboardInterrupt):
                print("\n PIN entry cancelled by user")
                raise ValueError("PIN entry cancelled")
            except Exception as e:
                print(f" Error getting PIN: {e}")
                if attempt == max_attempts - 1:
                    raise

        raise ValueError("PIN is required for authentication - max attempts exceeded")

    def update_pin(self) -> bool:
        """Update or change the saved PIN with better input handling"""
        print("\n" + "=" * 50)
        print("UPDATE TRADING PIN")
        print("=" * 50)
        print("This will update your saved Fyers trading PIN.")

        try:
            # Get current PIN for verification (if exists)
            if self.pin:
                print("Current PIN is configured.")
                verify_current = input("Verify current PIN first? (y/n) [y]: ").strip().lower()

                if verify_current != 'n':
                    current_pin = self._secure_input("Enter current PIN: ")
                    if current_pin != self.pin:
                        print(" Current PIN verification failed!")
                        return False
                    print(" Current PIN verified")

            # Get new PIN
            new_pin = self._secure_input("Enter new PIN: ")

            if not new_pin:
                print(" PIN cannot be empty")
                return False

            if not new_pin.isdigit():
                print(" PIN must contain only numbers")
                return False

            if len(new_pin) < 4:
                print(" PIN must be at least 4 digits")
                return False

            # Confirm new PIN
            confirm_pin = self._secure_input("Confirm new PIN: ")

            if new_pin != confirm_pin:
                print(" PINs do not match!")
                return False

            # Save new PIN
            if self.save_to_env('FYERS_PIN', new_pin):
                self.pin = new_pin
                print(" PIN updated successfully!")
                return True
            else:
                print(" Failed to save new PIN")
                return False

        except Exception as e:
            print(f" Error updating PIN: {e}")
            return False

    def update_pin_simple(self) -> bool:
        """Simple PIN update method using regular input (fallback)"""
        print("\n" + "=" * 50)
        print("UPDATE TRADING PIN (Simple Mode)")
        print("=" * 50)
        print("️  PIN will be visible on screen in this mode")

        try:
            new_pin = input("Enter new PIN: ").strip()

            if not new_pin:
                print(" PIN cannot be empty")
                return False

            if not new_pin.isdigit():
                print(" PIN must contain only numbers")
                return False

            if len(new_pin) < 4:
                print(" PIN must be at least 4 digits")
                return False

            confirm_pin = input("Confirm new PIN: ").strip()

            if new_pin != confirm_pin:
                print(" PINs do not match!")
                return False

            if self.save_to_env('FYERS_PIN', new_pin):
                self.pin = new_pin
                print(" PIN updated successfully")
                return True
            else:
                print(" Error saving PIN")
                return False

        except Exception as e:
            print(f" Error updating PIN: {e}")
            return False

    def get_app_id_hash(self) -> str:
        """Generate app_id_hash for API calls"""
        app_id = f"{self.client_id}:{self.secret_key}"
        return hashlib.sha256(app_id.encode()).hexdigest()

    def generate_auth_url(self) -> str:
        """Generate authorization URL for Fyers login"""
        try:
            params = {
                'client_id': self.client_id,
                'redirect_uri': self.redirect_uri,
                'response_type': 'code',
                'state': 'sample_state'
            }

            url = f"{self.auth_url}?" + "&".join([f"{k}={v}" for k, v in params.items()])
            logger.debug("Generated auth URL successfully")
            return url

        except Exception as e:
            logger.error(f"Error generating auth URL: {e}")
            return None

    def get_access_token_from_auth_code(self, auth_code: str) -> Optional[str]:
        """Exchange auth code for access token (SEBI: no refresh tokens issued)"""
        try:
            logger.info("Exchanging auth code for access token...")

            headers = {"Content-Type": "application/json"}

            data = {
                "grant_type": "authorization_code",
                "appIdHash": self.get_app_id_hash(),
                "code": auth_code
            }

            response = requests.post(self.token_url, headers=headers, json=data, timeout=30)

            try:
                response_data = response.json()
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON response from token API: {e}")
                logger.debug(f"Response content: {response.text}")
                logger.error(f"Token exchange failed: Invalid response format")
                return None

            if response.status_code == 200 and response_data.get('s') == 'ok':
                access_token = response_data.get('access_token')
                logger.info("Successfully obtained access token from auth code")
                return access_token
            else:
                error_msg = response_data.get('message', 'Unknown error')
                error_code = response_data.get('code', 'Unknown')
                logger.error(f"Token exchange failed: {error_msg} (Code: {error_code})")
                return None

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error during token exchange: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during token exchange: {e}")
            return None

    def is_token_valid(self, access_token: str) -> bool:
        """Check if access token is still valid"""
        if not access_token or not self.client_id:
            return False

        try:
            headers = {'Authorization': f"{self.client_id}:{access_token}"}
            response = requests.get(self.profile_url, headers=headers, timeout=10)

            if response.status_code == 200:
                result = response.json()
                is_valid = result.get('s') == 'ok'
                logger.debug(f"Token validation result: {'valid' if is_valid else 'invalid'}")
                return is_valid
            else:
                logger.debug(f"Token validation failed with status: {response.status_code}")
                return False

        except Exception as e:
            logger.debug(f"Token validation error: {e}")
            return False

    def get_valid_access_token(self) -> Optional[str]:
        """Get a valid access token, enforcing SEBI daily 2FA requirement.

        Per SEBI guidelines (effective April 1, 2026): refresh tokens are
        discontinued. Full re-authentication via 2FA is required each trading day.
        """
        try:
            today = datetime.date.today().isoformat()
            last_auth_date = os.environ.get('FYERS_LAST_AUTH_DATE', '')

            # SEBI daily 2FA: if today's auth hasn't been completed, check first whether
            # an existing token (e.g. manually placed in .env before batch run) is valid.
            # This avoids forcing interactive re-auth when a good token is already present.
            if last_auth_date != today:
                if self.access_token and self.is_token_valid(self.access_token):
                    logger.info(
                        f"Existing access token is valid. Updating last auth date to {today} "
                        f"(was '{last_auth_date or 'never'}')."
                    )
                    self.save_to_env('FYERS_LAST_AUTH_DATE', today)
                    return self.access_token

                logger.info(
                    f"SEBI daily 2FA required: last auth date was '{last_auth_date or 'never'}', "
                    f"today is {today}. Initiating full re-authentication."
                )
                print(f"\n{'=' * 60}")
                print("SEBI DAILY 2FA AUTHENTICATION REQUIRED")
                print("=" * 60)
                print("Per SEBI guidelines (effective April 1, 2026), daily 2FA is mandatory.")
                print("Refresh tokens are no longer supported.")
                print(f"Last authenticated: {last_auth_date or 'never'}")
                access_token = self.setup_full_authentication()
                if access_token:
                    self.save_to_env('FYERS_LAST_AUTH_DATE', today)
                    logger.info(f"SEBI daily 2FA completed successfully for {today}")
                return access_token

            # Same-day: validate existing token
            if self.access_token and self.is_token_valid(self.access_token):
                logger.info("Current access token is still valid")
                return self.access_token

            # Token expired mid-session — full re-auth required (no refresh tokens per SEBI)
            logger.info("Access token is invalid or expired. Full re-authentication required (SEBI: no refresh tokens).")
            access_token = self.setup_full_authentication()
            if access_token:
                self.save_to_env('FYERS_LAST_AUTH_DATE', today)
            return access_token

        except Exception as e:
            logger.error(f"Error getting valid access token: {e}")
            return None

    def setup_cli_authentication(self) -> Optional[str]:
        """Complete daily 2FA authentication via command line (no browser)"""
        try:
            print("\n" + "=" * 70)
            print("FYERS API DAILY 2FA AUTHENTICATION (CLI-ONLY)")
            print("=" * 70)
            print("Complete authentication entirely in the terminal.")
            print("No browser required - all input via command line.")
            print("Refresh tokens are disabled per SEBI regulations (effective April 1, 2026).")

            if not all([self.client_id, self.secret_key]):
                print(" Missing CLIENT_ID or SECRET_KEY in environment variables")
                return None

            # Generate auth URL (for manual reference if needed)
            print(f"\n STEP 1: LOGIN INFORMATION")
            print("=" * 70)
            auth_url = self.generate_auth_url()
            if not auth_url:
                print(" Failed to generate authentication URL")
                return None

            print(f"\nIf needed, the authentication URL is:")
            print(f"{auth_url}")

            # Collect credentials from command line
            print(f"\n STEP 2: ENTER YOUR CREDENTIALS")
            print("=" * 70)

            # Phone number / email
            print(f"\nEnter your Fyers login credentials:")
            while True:
                phone_or_email = input(" Phone Number / Email: ").strip()
                if phone_or_email:
                    break
                print(" Cannot be empty. Please enter your phone number or email.")

            # Password
            print(f"\nEnter your Fyers password:")
            password = self._secure_input(" Password: ")
            if not password:
                print(" Password cannot be empty")
                return None

            # OTP - Trigger sending
            print(f"\n STEP 3: SEND & VERIFY OTP")
            print("=" * 70)
            print("Sending OTP to your registered phone number...")

            request_id = self._send_otp(phone_or_email, password)
            if not request_id:
                print(" Failed to send OTP. Please check your credentials and try again.")
                return None

            print("OTP has been sent to your registered phone number.")
            print("Check your SMS and enter the OTP below.")

            while True:
                otp = input(" Enter OTP (6 digits): ").strip()
                if otp.isdigit() and len(otp) == 6:
                    break
                print(" OTP must be 6 digits. Please try again.")

            # Trading PIN
            print(f"\n STEP 4: ENTER TRADING PIN")
            print("=" * 70)
            print("Enter your Fyers trading PIN (4-6 digits).")

            max_pin_attempts = 3
            for attempt in range(max_pin_attempts):
                pin = self._secure_input(" Trading PIN: ")

                if not pin.isdigit():
                    print(" PIN must contain only numbers")
                    continue

                if len(pin) < 4 or len(pin) > 10:
                    print(" PIN must be 4-10 digits")
                    continue

                # Verify PIN
                confirm_pin = self._secure_input(" Confirm Trading PIN: ")
                if pin != confirm_pin:
                    print(" PINs do not match")
                    continue

                break
            else:
                print(" Maximum PIN attempts exceeded")
                return None

            # Save PIN for future use
            print(f"\n Saving trading PIN...")
            if self.save_to_env('FYERS_PIN', pin):
                self.pin = pin
                print(f" ✓ Trading PIN saved to .env file")

            # Get authorization code via CLI
            print(f"\n STEP 5: GET AUTHORIZATION CODE")
            print("=" * 70)
            print("Attempting to obtain authorization code using direct API...")

            # Attempt direct API authentication with the OTP we just verified
            auth_code = self._verify_otp_and_get_authcode(
                phone_or_email, otp, pin, request_id
            )

            if not auth_code:
                print(" Direct API authentication failed.")
                print("\nFalling back to manual option:")
                print(f" Visit this URL in your browser: {auth_url}")
                print(" Login with the credentials you just provided")
                print(" Complete the 2FA process in the browser")
                print(" Copy the authorization code from the redirect URL")
                auth_code = input(" Paste the authorization code here: ").strip()

                if not auth_code:
                    print(" No authorization code provided")
                    return None

            # Exchange auth code for access token
            print(f"\n STEP 6: EXCHANGE CODE FOR TOKEN")
            print("=" * 70)
            print(" Exchanging authorization code for access token...")
            access_token = self.get_access_token_from_auth_code(auth_code)

            if not access_token:
                print(" Failed to obtain access token")
                return None

            print(" ✓ Access token obtained successfully!")

            # Save access token to .env
            print(f"\n STEP 7: SAVE TOKEN")
            print("=" * 70)
            if self.save_to_env('FYERS_ACCESS_TOKEN', access_token):
                print(f" ✓ Access token saved to .env file")

            # Verify the setup
            if self.is_token_valid(access_token):
                print(f"\n AUTHENTICATION SUCCESSFUL!")
                print("=" * 70)
                print(f" ✓ Access token is valid and ready to use")

                # Try to get profile info
                try:
                    headers = {'Authorization': f"{self.client_id}:{access_token}"}
                    response = requests.get(self.profile_url, headers=headers, timeout=10)

                    if response.status_code == 200:
                        result = response.json()
                        if result.get('s') == 'ok':
                            profile_data = result.get('data', {})
                            print(f" Account Name: {profile_data.get('name', 'Unknown')}")
                            print(f" Email: {profile_data.get('email', 'Unknown')}")
                            print(f" User ID: {profile_data.get('id', 'Unknown')}")
                except:
                    pass  # Profile fetch is optional

                print(f"\n Ready to run strategy: python main.py run")
                return access_token
            else:
                print(f" Token validation failed after setup")
                return None

        except KeyboardInterrupt:
            print(f"\n\n Authentication cancelled by user")
            return None
        except Exception as e:
            print(f" CLI authentication setup failed: {e}")
            logger.exception("CLI authentication setup error")
            return None

    def _send_otp(self, phone_or_email: str, password: str) -> Optional[str]:
        """Send OTP to user's registered phone number"""
        try:
            logger.info(f"Sending OTP to {phone_or_email}...")

            headers = {"Content-Type": "application/json"}

            # This endpoint sends OTP to the registered phone
            otp_url = "https://api-t1.fyers.in/api/v3/send-login-otp"

            data = {
                "fy_id": phone_or_email,
                "password": password,
                "appIdHash": self.get_app_id_hash()
            }

            response = requests.post(otp_url, headers=headers, json=data, timeout=30)

            try:
                response_data = response.json()
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON response from OTP API: {e}")
                logger.debug(f"Response content: {response.text}")
                print(f" Network error: Invalid response format from server")
                return None

            if response.status_code == 200 and response_data.get('s') == 'ok':
                logger.info("OTP sent successfully")
                return response_data.get('request_id') or 'otp_sent'
            else:
                error_msg = response_data.get('message', 'Failed to send OTP')
                logger.error(f"OTP sending failed: {error_msg}")
                print(f" Error: {error_msg}")
                return None

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error sending OTP: {e}")
            print(f" Network error: {e}")
            return None
        except Exception as e:
            logger.error(f"Error sending OTP: {e}")
            print(f" Error: {e}")
            return None

    def _verify_otp_and_get_authcode(
        self, phone_or_email: str, otp: str, pin: str, request_id: str = None
    ) -> Optional[str]:
        """Verify OTP and PIN, then get authorization code"""
        try:
            logger.info("Verifying OTP and PIN...")

            headers = {"Content-Type": "application/json"}

            # This endpoint verifies OTP and PIN to get auth code
            verify_url = "https://api-t1.fyers.in/api/v3/verify-otp"

            data = {
                "fy_id": phone_or_email,
                "otp": otp,
                "pin": pin,
                "appIdHash": self.get_app_id_hash()
            }

            if request_id:
                data["request_id"] = request_id

            response = requests.post(verify_url, headers=headers, json=data, timeout=30)

            try:
                response_data = response.json()
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON response from verify OTP API: {e}")
                logger.debug(f"Response content: {response.text}")
                print(f" Network error: Invalid response format from server")
                return None

            if response.status_code == 200 and response_data.get('s') == 'ok':
                auth_code = response_data.get('auth_code')
                if auth_code:
                    logger.info("OTP and PIN verified successfully")
                    return auth_code
                else:
                    logger.error("No auth code in response")
                    print(" Error: No authorization code received")
                    return None
            else:
                error_msg = response_data.get('message', 'OTP verification failed')
                logger.error(f"OTP verification failed: {error_msg}")
                print(f" Error: {error_msg}")
                return None

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error verifying OTP: {e}")
            print(f" Network error: {e}")
            return None
        except Exception as e:
            logger.error(f"Error verifying OTP: {e}")
            print(f" Error: {e}")
            return None

    def _attempt_direct_authentication(
        self, phone_or_email: str, password: str, otp: str, pin: str
    ) -> Optional[str]:
        """Attempt direct API authentication without browser"""
        try:
            logger.info("Attempting direct Fyers API authentication...")

            # Step 1: Send OTP
            print("  Sending OTP to your registered phone number...")
            request_id = self._send_otp(phone_or_email, password)

            if not request_id:
                logger.error("Failed to send OTP")
                return None

            print("  OTP sent successfully!")

            # Step 2: Verify OTP and PIN to get auth code
            print("  Verifying OTP and PIN...")
            auth_code = self._verify_otp_and_get_authcode(phone_or_email, otp, pin, request_id)

            if not auth_code:
                logger.error("Failed to verify OTP and PIN")
                return None

            print("  Authorization code obtained!")
            return auth_code

        except Exception as e:
            logger.debug(f"Direct authentication attempt failed: {e}")
            print(f" Direct authentication failed: {e}")
            return None

    def setup_browser_authentication(self, port: int = 5000) -> Optional[str]:
        """Complete daily 2FA authentication via auto-opened browser with redirect capture"""
        try:
            print("\n" + "=" * 70)
            print("FYERS API DAILY 2FA AUTHENTICATION (BROWSER-BASED)")
            print("=" * 70)
            print("A browser window will open automatically for authentication.")
            print("Refresh tokens are disabled per SEBI regulations (effective April 1, 2026).")

            if not all([self.client_id, self.secret_key]):
                print(" Missing CLIENT_ID or SECRET_KEY in environment variables")
                return None

            # Update redirect URI to use localhost
            self.redirect_uri = f"http://localhost:{port}/callback"

            # Setup local server to capture redirect
            print(f"\n Starting local authorization server on port {port}...")
            try:
                server = HTTPServer(('localhost', port), AuthCodeRedirectHandler)
                server_thread = threading.Thread(target=server.handle_request, daemon=True)
                server_thread.start()
                time.sleep(0.5)  # Give server time to start
            except OSError as e:
                print(f"️  Could not start local server on port {port}: {e}")
                print(" Falling back to manual auth code entry...")
                return self.setup_full_authentication()

            # Generate auth URL with localhost redirect
            print(f" Opening browser for authentication...")
            auth_url = self.generate_auth_url()
            if not auth_url:
                print(" Failed to generate authentication URL")
                server.server_close()
                return None

            print(f" Auth URL: {auth_url}")

            # Open browser automatically
            browser_opened = webbrowser.open(auth_url)
            if browser_opened:
                print(" Browser opened successfully")
            else:
                print("⚠  Could not open browser automatically")
                print(" Please open this URL manually in your browser:")
                print(f" {auth_url}")

            print(f"\n Waiting for authentication...")
            print(f"1. Enter your phone number when prompted")
            print(f"2. Verify the OTP sent to your phone")
            print(f"3. Enter your trading PIN")
            print(f"4. The authorization code will be captured automatically")

            # Wait for redirect callback with timeout
            timeout = 300  # 5 minutes
            if AuthCodeRedirectHandler.redirect_received.wait(timeout):
                auth_code = AuthCodeRedirectHandler.auth_code
                print(f"\n ✓ Authorization code received!")
                server.server_close()
            else:
                print(f"\n ✗ Timeout waiting for authorization code")
                server.server_close()
                return None

            # Exchange auth code for access token
            print(f" Exchanging authorization code for access token...")
            access_token = self.get_access_token_from_auth_code(auth_code)

            if not access_token:
                print(" Failed to obtain access token")
                return None

            print(" ✓ Access token obtained successfully!")

            # Save access token to .env
            print(f"\n Saving access token...")
            if self.save_to_env('FYERS_ACCESS_TOKEN', access_token):
                print(f" ✓ Access token saved to .env file")

            # Verify the setup
            if self.is_token_valid(access_token):
                print(f"\n AUTHENTICATION SUCCESSFUL!")
                print(f" ✓ Access token is valid and ready to use")

                # Try to get profile info
                try:
                    headers = {'Authorization': f"{self.client_id}:{access_token}"}
                    response = requests.get(self.profile_url, headers=headers, timeout=10)

                    if response.status_code == 200:
                        result = response.json()
                        if result.get('s') == 'ok':
                            profile_data = result.get('data', {})
                            print(f" Account: {profile_data.get('name', 'Unknown')}")
                            print(f" Email: {profile_data.get('email', 'Unknown')}")
                except:
                    pass  # Profile fetch is optional

                return access_token
            else:
                print(f" Token validation failed after setup")
                return None

        except Exception as e:
            print(f" Browser authentication setup failed: {e}")
            logger.exception("Browser authentication setup error")
            return None

    def setup_full_authentication(self) -> Optional[str]:
        """Complete daily 2FA authentication flow (SEBI-compliant, no refresh tokens)"""
        try:
            print("\n" + "=" * 70)
            print("FYERS API DAILY 2FA AUTHENTICATION (SEBI-COMPLIANT)")
            print("=" * 70)
            print("Refresh tokens are disabled per SEBI regulations (effective April 1, 2026).")
            print("You must complete this 2FA authentication every trading day.")

            if not all([self.client_id, self.secret_key]):
                print(" Missing CLIENT_ID or SECRET_KEY in environment variables")
                return None

            # Generate auth URL
            auth_url = self.generate_auth_url()
            if not auth_url:
                print(" Failed to generate authentication URL")
                return None

            print(f"\n AUTHENTICATION STEPS:")
            print(f"1. Open this URL in your web browser:")
            print(f"   {auth_url}")
            print(f"\n2. Log in to your Fyers account and complete 2FA")
            print(f"3. Copy the authorization code from the redirect URL")
            print(f"\n The redirect URL will look like:")
            print(f"   {self.redirect_uri}?code=YOUR_AUTH_CODE&state=sample_state")

            # Get auth code from user
            print(f"\n" + "=" * 50)
            auth_code = input(" Enter the authorization code: ").strip()

            if not auth_code:
                print(" No authorization code provided")
                return None

            # Exchange auth code for access token
            print(" Exchanging authorization code for access token...")
            access_token = self.get_access_token_from_auth_code(auth_code)

            if not access_token:
                print(" Failed to obtain access token")
                return None

            print(" Access token obtained successfully!")

            # Save access token to .env
            print(f"\n Saving access token...")
            if self.save_to_env('FYERS_ACCESS_TOKEN', access_token):
                print(f" Saved: Access Token")

            # Verify the setup
            if self.is_token_valid(access_token):
                print(f"\n AUTHENTICATION SUCCESSFUL!")
                print(f" Access token is valid and ready to use")

                # Try to get profile info
                try:
                    headers = {'Authorization': f"{self.client_id}:{access_token}"}
                    response = requests.get(self.profile_url, headers=headers, timeout=10)

                    if response.status_code == 200:
                        result = response.json()
                        if result.get('s') == 'ok':
                            profile_data = result.get('data', {})
                            print(f" Account: {profile_data.get('name', 'Unknown')}")
                            print(f" Email: {profile_data.get('email', 'Unknown')}")
                except:
                    pass  # Profile fetch is optional

                return access_token
            else:
                print(f" Token validation failed after setup")
                return None

        except Exception as e:
            print(f" Authentication setup failed: {e}")
            logger.exception("Full authentication setup error")
            return None

    def get_profile_info(self, access_token: str = None) -> Dict[str, Any]:
        """Get user profile information"""
        try:
            token_to_use = access_token or self.access_token
            if not token_to_use:
                return {'error': 'No access token available'}

            headers = {'Authorization': f"{self.client_id}:{token_to_use}"}
            response = requests.get(self.profile_url, headers=headers, timeout=10)

            if response.status_code == 200:
                result = response.json()
                if result.get('s') == 'ok':
                    return result.get('data', {})
                else:
                    return {'error': result.get('message', 'API error')}
            else:
                return {'error': f'HTTP {response.status_code}'}

        except Exception as e:
            return {'error': str(e)}


# Convenience functions for main.py
def setup_auth_only():
    """SEBI-compliant daily 2FA authentication setup with multiple methods"""
    print("=" * 80)
    print("FYERS API DAILY 2FA AUTHENTICATION SETUP (SEBI-COMPLIANT)")
    print("=" * 80)

    try:
        # Check if we already have credentials in environment
        existing_client_id = os.environ.get('FYERS_CLIENT_ID')
        existing_secret_key = os.environ.get('FYERS_SECRET_KEY')

        if existing_client_id and existing_secret_key:
            print("Found existing API credentials in environment")
            print("This will perform FULL re-authentication (not refresh)")

            print("\n" + "=" * 70)
            print("AUTHENTICATION METHOD SELECTION")
            print("=" * 70)
            print("\n1. CLI-Only (No Browser) - Complete in terminal")
            print("   - Provide phone, password, OTP, PIN via command line")
            print("   - Paste authorization code from browser redirect\n")
            print("2. Browser-Based (Recommended) - Browser opens automatically")
            print("   - Browser opens with login page")
            print("   - Authorization code captured automatically\n")
            print("3. Manual Code Entry - Copy/paste from browser")
            print("   - Open URL manually in browser")
            print("   - Copy authorization code and paste here\n")

            choice = input("Select authentication method (1/2/3) [default: 1]: ").strip()

            confirm = input("Proceed with full re-authentication? (y/n) [y]: ").strip().lower()
            if confirm == 'n':
                print("Authentication setup cancelled")
                return False

            auth_manager = FyersAuthManager()
            auth_manager.client_id = existing_client_id
            auth_manager.secret_key = existing_secret_key

            # Choose authentication method
            if choice == '2':
                print("\nStarting browser-based authentication flow...")
                access_token = auth_manager.setup_browser_authentication()
            elif choice == '3':
                print("\nStarting manual authentication flow...")
                access_token = auth_manager.setup_full_authentication()
            else:
                print("\nStarting CLI-only authentication flow...")
                access_token = auth_manager.setup_cli_authentication()

            if access_token:
                print("\nAuthentication successful!")
                return True
            else:
                print("\nAuthentication setup failed!")
                return False

        # Manual setup if no credentials exist
        print("\n" + "=" * 50)
        print("MANUAL CREDENTIAL SETUP")
        print("=" * 50)

        print("Please enter your Fyers API credentials:")
        print("(Get these from: https://myapi.fyers.in/dashboard)")

        while True:
            client_id = input("\nEnter your Fyers Client ID: ").strip()
            if client_id:
                break
            print("Client ID cannot be empty")

        while True:
            secret_key = input("Enter your Fyers Secret Key: ").strip()
            if secret_key:
                break
            print("Secret Key cannot be empty")

        redirect_uri = input("Enter Redirect URI (press Enter for default): ").strip()
        if not redirect_uri:
            redirect_uri = "https://trade.fyers.in/api-login/redirect-to-app"

        # Save basic credentials
        auth_manager = FyersAuthManager()
        auth_manager.save_to_env('FYERS_CLIENT_ID', client_id)
        auth_manager.save_to_env('FYERS_SECRET_KEY', secret_key)
        auth_manager.save_to_env('FYERS_REDIRECT_URI', redirect_uri)

        # Update manager with new credentials
        auth_manager.client_id = client_id
        auth_manager.secret_key = secret_key
        auth_manager.redirect_uri = redirect_uri

        # Choose authentication method
        print("\n" + "=" * 70)
        print("AUTHENTICATION METHOD SELECTION")
        print("=" * 70)
        print("\n1. CLI-Only (No Browser) - Complete in terminal")
        print("   - Provide phone, password, OTP, PIN via command line")
        print("   - Paste authorization code from browser redirect\n")
        print("2. Browser-Based (Recommended) - Browser opens automatically")
        print("   - Browser opens with login page")
        print("   - Authorization code captured automatically\n")
        print("3. Manual Code Entry - Copy/paste from browser")
        print("   - Open URL manually in browser")
        print("   - Copy authorization code and paste here\n")

        choice = input("Select authentication method (1/2/3) [default: 1]: ").strip()

        # Perform authentication
        if choice == '2':
            print("\nPerforming browser-based authentication...")
            access_token = auth_manager.setup_browser_authentication()
        elif choice == '3':
            print("\nPerforming manual authentication...")
            access_token = auth_manager.setup_full_authentication()
        else:
            print("\nPerforming CLI-only authentication...")
            access_token = auth_manager.setup_cli_authentication()

        if access_token:
            print("\nSEBI-compliant daily 2FA authentication setup completed!")
            print("Note: Refresh tokens are disabled per SEBI regulations.")
            return True
        else:
            print("\nAuthentication setup failed!")
            return False

    except KeyboardInterrupt:
        print("\n\nAuthentication setup cancelled by user")
        return False
    except Exception as e:
        print(f"\nAuthentication setup error: {e}")
        logger.exception("Setup authentication error")
        return False


def authenticate_fyers(config_dict: dict) -> bool:
    """Handle Fyers authentication — SEBI daily 2FA (no refresh tokens)"""
    try:
        auth_manager = FyersAuthManager()

        # Get valid access token (SEBI: full re-auth required each trading day)
        access_token = auth_manager.get_valid_access_token()

        if access_token:
            # Update config with the valid token
            config_dict['fyers_config'].access_token = access_token
            logger.info("Fyers authentication successful")
            return True
        else:
            logger.error("Fyers authentication failed")
            return False

    except Exception as e:
        logger.error(f"Authentication error: {e}")
        return False


def test_authentication():
    """Test authentication without running strategies"""
    try:
        print("\n" + "=" * 60)
        print("FYERS AUTHENTICATION TEST")
        print("=" * 60)

        auth_manager = FyersAuthManager()

        if not all([auth_manager.client_id, auth_manager.secret_key]):
            print("Missing API credentials")
            print("Run: python main.py auth")
            return False

        print(" Testing authentication...")

        # Test token validity
        access_token = auth_manager.get_valid_access_token()

        if not access_token:
            print("Authentication failed - no valid access token")
            return False

        print("Authentication successful!")

        # Test API call
        print("Testing API connection...")
        profile_info = auth_manager.get_profile_info(access_token)

        if 'error' in profile_info:
            print(f"Profile fetch failed: {profile_info['error']}")
        else:
            print("API connection successful!")
            print(f"Name: {profile_info.get('name', 'Unknown')}")
            print(f"Email: {profile_info.get('email', 'Unknown')}")
            print(f"User ID: {profile_info.get('id', 'Unknown')}")

        return True

    except Exception as e:
        print(f"Authentication test failed: {e}")
        logger.exception("Authentication test error")
        return False


def update_pin_only():
    """Update trading PIN only with improved error handling"""
    try:
        print("\n" + "=" * 60)
        print("UPDATE FYERS TRADING PIN")
        print("=" * 60)

        auth_manager = FyersAuthManager()

        print("Choose PIN update method:")
        print("1. Secure mode (PIN hidden) - Recommended")
        print("2. Simple mode (PIN visible) - Fallback option")

        choice = input("\nEnter choice (1/2) [default: 1]: ").strip()

        if choice == "2":
            success = auth_manager.update_pin_simple()
        else:
            success = auth_manager.update_pin()

        if success:
            print("\n PIN update completed successfully!")
            print("Your new PIN has been saved to the .env file")
        else:
            print("\nPIN update failed. Please try again.")

        return success

    except Exception as e:
        print(f"\nPIN update error: {e}")
        logger.exception("PIN update error")
        return False


def test_pin_input():
    """Test PIN input methods to see which works in your environment"""
    print("\n" + "=" * 60)
    print("PIN INPUT METHOD TESTING")
    print("=" * 60)

    # Test 1: getpass
    print("Testing secure input (getpass):")
    getpass_works = False
    try:
        test_pin = getpass.getpass("Enter test PIN (will be hidden): ")
        print(f"Secure input works! Entered: {'*' * len(test_pin)} ({len(test_pin)} digits)")
        getpass_works = True
    except Exception as e:
        print(f"Secure input failed: {e}")

    # Test 2: regular input
    print(f"\nTesting regular input:")
    regular_works = False
    try:
        test_pin = input("Enter test PIN (will be visible): ")
        print(f"Regular input works! Entered: {test_pin}")
        regular_works = True
    except Exception as e:
        print(f"Regular input failed: {e}")

    # Recommendations
    print(f"\nRECOMMENDATIONS:")
    if getpass_works:
        print("Use secure mode (option 1) for PIN operations")
    elif regular_works:
        print("Use simple mode (option 2) for PIN operations")
        print("   Note: PIN will be visible on screen")
    else:
        print("Both input methods failed - check your environment")

    return getpass_works, regular_works


def show_environment_info():
    """Show information about the current environment"""
    print("\n" + "=" * 60)
    print("ENVIRONMENT INFORMATION")
    print("=" * 60)

    print(f"Python Version: {sys.version}")
    print(f"Platform: {sys.platform}")
    print(f"Interactive Terminal: {sys.stdin.isatty()}")

    # Check if running in various environments
    environments = []
    if 'jupyter' in sys.modules or 'IPython' in sys.modules:
        environments.append("Jupyter/IPython")
    if 'VSCODE_PID' in os.environ:
        environments.append("VS Code")
    if 'PYCHARM_HOSTED' in os.environ:
        environments.append("PyCharm")
    if os.environ.get('TERM_PROGRAM') == 'vscode':
        environments.append("VS Code Terminal")

    if environments:
        print(f"Detected Environment: {', '.join(environments)}")
    else:
        print(f"Environment: Standard Terminal")

    print(f"\nNote: getpass (secure input) may not work in some IDEs or notebook environments")


# Quick test function
if __name__ == "__main__":
    print("Enhanced Fyers Authentication Helper - Standalone Test")
    print("=" * 60)

    # Show environment info
    show_environment_info()

    # Test PIN input methods
    print("\nTesting PIN input methods...")
    test_pin_input()

    # Basic functionality test
    print(f"\nTesting authentication manager...")
    try:
        auth_manager = FyersAuthManager()
        print(f"FyersAuthManager created successfully")
        print(f"Client ID configured: {'Yes' if auth_manager.client_id else 'No'}")
        print(f"Secret Key configured: {'Yes' if auth_manager.secret_key else 'No'}")
        print(f"Access Token configured: {'Yes' if auth_manager.access_token else 'No'}")
        print(f"Note: Refresh tokens disabled per SEBI regulations (April 1, 2026)")

        # Test .env file operations
        test_key = "TEST_KEY_" + str(int(time.time()))
        test_value = "test_value_123"

        if auth_manager.save_to_env(test_key, test_value):
            print(f" .env file operations work correctly")
            # Clean up test key
            try:
                env_file = '.env'
                if os.path.exists(env_file):
                    with open(env_file, 'r') as f:
                        lines = f.readlines()
                    with open(env_file, 'w') as f:
                        for line in lines:
                            if not line.startswith(test_key):
                                f.write(line)
            except:
                pass
        else:
            print(f" .env file operations failed")

    except Exception as e:
        print(f" Authentication manager test failed: {e}")

    print(f"\n To setup full authentication, run:")
    print(f"   python main.py auth")