# utils/enhanced_auth_helper.py

"""
Enhanced Fyers Authentication Helper for ORB Trading Strategy
CLI-only SEBI-compliant daily 2FA authentication (no browser flow).
"""

import base64
import hashlib
import re
import requests
import logging
import os
import json
import getpass
import sys
import time
import datetime
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class FyersAuthManager:
    """Fyers authentication manager — SEBI-compliant daily 2FA (no refresh tokens)"""

    def __init__(self):
        self.client_id = os.environ.get('FYERS_CLIENT_ID')
        self.secret_key = os.environ.get('FYERS_SECRET_KEY')
        self.redirect_uri = os.environ.get('FYERS_REDIRECT_URI', "https://trade.fyers.in/api-login/redirect-to-app")
        self.access_token = os.environ.get('FYERS_ACCESS_TOKEN')
        self.pin = os.environ.get('FYERS_PIN')
        # Fyers login username (Client ID like "XK00123"). This is distinct from
        # FYERS_CLIENT_ID which identifies the registered App (e.g. "XXXXXX-100").
        self.fy_id = os.environ.get('FYERS_FY_ID')

        # API endpoints
        self.auth_url = "https://api-t1.fyers.in/api/v3/generate-authcode"
        self.token_url = "https://api-t1.fyers.in/api/v3/validate-authcode"
        self.profile_url = "https://api-t1.fyers.in/api/v3/profile"

        # Shared HTTP session for the login flow. Cookies set by the vagator
        # service (e.g. CSRF / session) must persist across send_otp →
        # verify_otp → verify_pin → token. Also matches the browser-like
        # request pattern the vagator endpoints expect.
        self._login_session: Optional[requests.Session] = None

    def _parse_json_response(self, response_text: str) -> dict:
        """Parse JSON response, handling extra data after JSON object"""
        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            # Try to extract valid JSON if there's extra data
            if "Extra data" in str(e):
                # Find the end of the JSON object by tracking braces
                brace_count = 0
                in_string = False
                escape_next = False

                for i, char in enumerate(response_text):
                    if escape_next:
                        escape_next = False
                        continue

                    if char == '\\':
                        escape_next = True
                        continue

                    if char == '"' and not escape_next:
                        in_string = not in_string
                        continue

                    if not in_string:
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                # Found the end of the JSON object
                                return json.loads(response_text[:i + 1])

            # If extraction failed, raise the original error
            raise

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
                response_data = self._parse_json_response(response.text)
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
                result = self._parse_json_response(response.text)
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
                access_token = self.setup_cli_authentication()
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
            access_token = self.setup_cli_authentication()
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

            # Fyers Client ID (login username, e.g. "XK00123"). This is NOT the
            # app's FYERS_CLIENT_ID and NOT a phone number or email.
            print(f"\nEnter your Fyers login credentials:")
            print(" Your Fyers Client ID is your login username (e.g. 'XK00123').")
            print(" It is the same ID you use at https://login.fyers.in/.")
            default_fy_id = self.fy_id or ''

            # Fyers login flow is OTP + PIN based; no password is required.
            # OTP - Trigger sending. Retry on failure so a typo'd Client ID
            # doesn't force the user to restart the whole flow.
            print(f"\n STEP 3: SEND & VERIFY OTP")
            print("=" * 70)

            max_send_attempts = 3
            request_id = None
            fy_id = None
            for attempt in range(max_send_attempts):
                default_hint = f" [{default_fy_id}]" if default_fy_id else ""
                while True:
                    candidate = input(f" Fyers Client ID{default_hint}: ").strip() or default_fy_id
                    if candidate:
                        break
                    print(" Cannot be empty. Please enter your Fyers Client ID.")

                print("Sending OTP to your registered phone number...")
                request_id = self._send_otp(candidate)
                if request_id:
                    fy_id = candidate
                    break

                remaining = max_send_attempts - attempt - 1
                if remaining > 0:
                    print(f" Failed to send OTP. {remaining} attempt(s) left — try a different Client ID.")
                    # Don't auto-fill the bad value as the next default
                    default_fy_id = ''
                else:
                    print(" Failed to send OTP after multiple attempts. Please check your Fyers Client ID and try again.")
                    return None

            # Save fy_id for future runs now that we know it's valid
            if fy_id != self.fy_id:
                self.save_to_env('FYERS_FY_ID', fy_id)
                self.fy_id = fy_id

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
                fy_id, otp, pin, request_id
            )

            if not auth_code:
                print(" Direct API authentication failed.")
                print(" Check your Fyers Client ID, OTP, and PIN and try again.")
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
                        result = self._parse_json_response(response.text)
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

    @staticmethod
    def _b64(value: str) -> str:
        """Base64-encode a string as required by Fyers login API"""
        return base64.b64encode(value.encode()).decode()

    def _get_login_session(self) -> requests.Session:
        """Return a lazily-initialised ``requests.Session`` for the login flow.

        The vagator login endpoints rely on cookies (CSRF / session) set by
        the initial ``send_login_otp`` response. Using a shared session keeps
        those cookies across the subsequent ``verify_otp`` / ``verify_pin`` /
        ``token`` calls.
        """
        if self._login_session is None:
            session = requests.Session()
            # Match the header set used by the widely-deployed reference
            # implementation (https://github.com/tkanhe/fyers-api-access-token-v3).
            # Notable omissions: Origin / Referer / Content-Type. In practice
            # those extra headers cause Fyers' edge to reject the request
            # with the generic "invalid request" (-1025) response.
            session.headers.update({
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": "en-US,en;q=0.9",
                "Content-Type": "application/json",
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
            })
            self._login_session = session
        return self._login_session

    def _reset_login_session(self) -> None:
        """Drop any cached cookies/state from the login session."""
        if self._login_session is not None:
            try:
                self._login_session.close()
            except Exception:
                pass
        self._login_session = None

    def _post_login_api(
        self,
        url: str,
        payload: dict,
        step: str,
        *,
        extra_headers: Optional[Dict[str, str]] = None,
        suppress_error_print: bool = False,
    ) -> Optional[dict]:
        """POST to a Fyers login-flow endpoint and return the parsed JSON body, or None on failure.

        Sends the body as a raw JSON string via ``data=`` (mirroring the
        reference implementation) so we don't force a ``Content-Type`` header
        that Fyers' edge has been observed to reject with -1025.
        """
        session = self._get_login_session()
        # Compact separators (no whitespace) match the byte-for-byte format of
        # the widely-used reference client. Fyers' edge has been observed to
        # reject pretty-printed JSON with -1025 "invalid request".
        body = json.dumps(payload, separators=(',', ':'))

        logger.debug(f"{step} → POST {url} | headers={dict(session.headers)} | body={body}")
        try:
            response = session.post(
                url,
                data=body,
                headers=extra_headers or None,
                timeout=30,
            )
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error during {step}: {e}")
            print(f" Network error: {e}")
            return None

        logger.debug(f"{step} ← HTTP {response.status_code} | body={response.text[:1000]}")
        try:
            response_data = self._parse_json_response(response.text)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response from {step} API: {e}")
            logger.error(f"Response status: {response.status_code}, content: {response.text[:500]}")
            print(f" Invalid response from server during {step} (HTTP {response.status_code})")
            return None

        if response.status_code == 200 and response_data.get('s') == 'ok':
            return response_data

        error_msg = response_data.get('message') or response_data.get('msg') or f"{step} failed"
        error_code = response_data.get('code', response.status_code)
        logger.error(f"{step} failed: {error_msg} (code: {error_code})")
        logger.error(f"{step} full response [{response.status_code}]: {response.text[:800]}")

        # Expose the error payload so callers can implement endpoint-version
        # fallbacks without re-parsing the HTTP response.
        response_data.setdefault('_error_code', error_code)
        response_data['_http_status'] = response.status_code

        if suppress_error_print:
            return response_data

        print(f" Error: {error_msg}")
        # Code -1025 is Fyers' generic "invalid request". Common causes:
        #   1. Fyers Client ID is wrong / has a typo, or a phone/email was entered
        #   2. The vagator endpoint version (v2 vs v3) changed server-side
        #   3. Extra headers (Origin/Referer/Content-Type) tripping the edge WAF
        if str(error_code) == '-1025' and step == 'send OTP':
            print(
                "   Hint: 'invalid request' from Fyers usually means the Fyers Client ID\n"
                "         is wrong. It must be your login username (e.g. 'XK00123') —\n"
                "         the same ID you use at https://login.fyers.in/. Double-check\n"
                "         for typos (letters vs digits, missing characters)."
            )
        return None

    def _send_otp(self, fy_id: str, password: str = None) -> Optional[str]:
        """Send OTP to the user's registered phone number.

        ``fy_id`` must be the Fyers Client ID / login username (e.g. ``"XK00123"``),
        NOT a phone number or email — the vagator endpoint rejects those with
        error code ``-1025`` ("invalid request").

        Tries the ``send_login_otp_v2`` endpoint on the vagator service first and
        falls back to ``send_login_otp_v3`` if Fyers rejects v2 with the generic
        -1025 "invalid request" (seen when Fyers rotates the supported version).
        The ``password`` argument is accepted for backward compatibility but is
        not used: the Fyers login flow is OTP + PIN based.
        """
        fy_id = (fy_id or "").strip()
        # Detect common mistake: user entered a 10-digit phone number instead of their
        # Fyers Client ID. Fail fast with a clear message rather than letting the
        # server return an opaque -1025.
        digits_only = fy_id.lstrip('+').replace(' ', '')
        if digits_only.isdigit() and len(digits_only) >= 10 and '@' not in fy_id:
            logger.error(f"Rejected fy_id '{fy_id}': looks like a phone number, not a Fyers Client ID")
            print(
                " Error: That looks like a phone number, but Fyers expects your\n"
                "        Fyers Client ID (login username, e.g. 'XK00123'). This is\n"
                "        the same ID you use to log in at https://login.fyers.in/."
            )
            return None
        if '@' in fy_id:
            logger.error(f"Rejected fy_id '{fy_id}': email is not accepted by the login endpoint")
            print(
                " Error: Email addresses are not accepted by the Fyers OTP endpoint.\n"
                "        Use your Fyers Client ID (login username, e.g. 'XK00123')."
            )
            return None
        # Detect common mistake: user entered the Fyers *App* Client ID (e.g.
        # "TMX3VZXIK5-200" or "ABCDEF-100") instead of their *login* username.
        # App Client IDs follow the pattern ALPHANUMERIC-DIGITS (the part set in
        # FYERS_CLIENT_ID / FYERS_SECRET_KEY environment variables).
        if re.match(r'^[A-Z0-9]+-\d{2,3}$', fy_id):
            logger.error(
                f"Rejected fy_id '{fy_id}': looks like an App Client ID, not a Fyers login username"
            )
            print(
                f" Error: '{fy_id}' looks like a Fyers App Client ID (the one used\n"
                "        for API access), NOT your login username.\n"
                "        Your login username is a short alphanumeric code like 'XK00123'\n"
                "        — the same ID you type at https://login.fyers.in/."
            )
            return None

        logger.info(f"Sending OTP for Fyers ID {fy_id}...")

        # Each send-OTP attempt starts a fresh login session so stale cookies
        # from a previous failed attempt don't poison the new flow.
        self._reset_login_session()

        payload = {
            "fy_id": self._b64(fy_id),
            "app_id": "2",
        }

        endpoints = [
            "https://api-t2.fyers.in/vagator/v2/send_login_otp_v2",
            "https://api-t2.fyers.in/vagator/v2/send_login_otp_v3",
            "https://api-t1.fyers.in/vagator/v2/send_login_otp_v2",
            "https://api-t1.fyers.in/vagator/v2/send_login_otp_v3",
        ]

        success_data: Optional[dict] = None
        last_error: Optional[dict] = None
        for otp_url in endpoints:
            # Ask _post_login_api to return the raw error body so we can
            # decide whether to try the next endpoint version before
            # surfacing the failure to the user.
            response_data = self._post_login_api(
                otp_url,
                payload,
                step="send OTP",
                suppress_error_print=True,
            )
            if response_data and response_data.get('s') == 'ok':
                success_data = response_data
                break

            last_error = response_data or {}
            error_code = str(last_error.get('_error_code', ''))
            http_status = last_error.get('_http_status')

            # Only fall through to the next endpoint version on signals that
            # point at an endpoint/version mismatch. Genuine validation errors
            # (bad Client ID, rate limits, etc.) should fail fast.
            if error_code != '-1025' and http_status != 404:
                break

            logger.info(
                f"send_login_otp rejected at {otp_url.rsplit('/', 1)[-1]} "
                f"(code {error_code or http_status}); trying next endpoint version."
            )

        if success_data is None:
            if last_error:
                err = (
                    last_error.get('message')
                    or last_error.get('msg')
                    or 'send OTP failed'
                )
                print(f" Error: {err}")
                if str(last_error.get('_error_code')) == '-1025':
                    print(
                        "   Hint: 'invalid request' from Fyers usually means the Fyers Client ID\n"
                        "         is wrong. It must be your login username (e.g. 'XK00123') —\n"
                        "         the same ID you use at https://login.fyers.in/. Double-check\n"
                        "         for typos (letters vs digits, missing characters)."
                    )
            return None

        request_key = (
            success_data.get('request_key')
            or (success_data.get('data') or {}).get('request_key')
        )
        if not request_key:
            logger.error(f"No request_key in send-OTP response: {success_data}")
            print(" Error: Server did not return a request key")
            return None

        logger.info("OTP sent successfully")
        return request_key

    def _verify_otp(self, request_key: str, otp: str) -> Optional[str]:
        """Verify the 6-digit OTP and return the next-step request_key"""
        logger.info("Verifying OTP...")

        verify_url = "https://api-t2.fyers.in/vagator/v2/verify_otp"
        payload = {"request_key": request_key, "otp": otp}

        response_data = self._post_login_api(verify_url, payload, step="verify OTP")
        if not response_data:
            return None

        next_key = (
            response_data.get('request_key')
            or (response_data.get('data') or {}).get('request_key')
        )
        if not next_key:
            logger.error(f"No request_key in verify-OTP response: {response_data}")
            print(" Error: Server did not return a request key after OTP verification")
            return None

        logger.info("OTP verified successfully")
        return next_key

    def _verify_pin(self, request_key: str, pin: str) -> Optional[str]:
        """Verify the trading PIN and return an interim access token"""
        logger.info("Verifying trading PIN...")

        verify_url = "https://api-t2.fyers.in/vagator/v2/verify_pin_v2"
        payload = {
            "request_key": request_key,
            "identity_type": "pin",
            "identifier": self._b64(pin),
        }

        response_data = self._post_login_api(verify_url, payload, step="verify PIN")
        if not response_data:
            return None

        data = response_data.get('data') or {}
        interim_token = data.get('access_token') or response_data.get('access_token')
        if not interim_token:
            logger.error(f"No access_token in verify-PIN response: {response_data}")
            print(" Error: Server did not return an interim access token after PIN verification")
            return None

        logger.info("PIN verified successfully")
        return interim_token

    def _get_auth_code(self, interim_token: str, fy_id: str) -> Optional[str]:
        """Exchange the interim access token for an authorization code"""
        logger.info("Requesting authorization code...")

        token_url = "https://api-t1.fyers.in/api/v3/token"
        # Fyers app_id is the part of the client_id before the "-<appType>" suffix.
        if self.client_id and '-' in self.client_id:
            app_id, app_type = self.client_id.rsplit('-', 1)
        else:
            app_id, app_type = self.client_id or '', '100'

        payload = {
            "fyers_id": fy_id,
            "app_id": app_id,
            "redirect_uri": self.redirect_uri,
            "appType": app_type,
            "code_challenge": "",
            "state": "sample_state",
            "scope": "",
            "nonce": "",
            "response_type": "code",
            "create_cookie": True,
        }

        # Reuse the login session so cookies set earlier in the flow carry
        # through to the token endpoint (mirrors the tkanhe reference).
        session = self._get_login_session()
        try:
            response = session.post(
                token_url,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {interim_token}",
                },
                data=json.dumps(payload),
                timeout=30,
                allow_redirects=False,
            )
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error requesting auth code: {e}")
            print(f" Network error: {e}")
            return None

        try:
            response_data = self._parse_json_response(response.text)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response from token API: {e}")
            logger.debug(f"Response status: {response.status_code}, content: {response.text[:500]}")
            print(f" Invalid response from server while fetching auth code (HTTP {response.status_code})")
            return None

        if response.status_code != 200 or response_data.get('s') != 'ok':
            error_msg = response_data.get('message') or response_data.get('msg') or 'auth code request failed'
            logger.error(f"Auth code request failed: {error_msg}")
            print(f" Error: {error_msg}")
            return None

        redirect_url = response_data.get('Url') or response_data.get('url') or ''
        match = re.search(r'[?&]auth_code=([^&]+)', redirect_url)
        if match:
            return match.group(1)

        data = response_data.get('data') or {}
        direct_code = data.get('auth_code') or response_data.get('auth_code')
        if direct_code:
            return direct_code

        logger.error(f"No auth_code found in token response: {response_data}")
        print(" Error: Server did not return an authorization code")
        return None

    def _verify_otp_and_get_authcode(
        self, fy_id: str, otp: str, pin: str, request_id: str = None
    ) -> Optional[str]:
        """Verify OTP and PIN, then get authorization code.

        Runs the full Fyers v3 login flow:
        ``verify_otp`` -> ``verify_pin_v2`` -> ``token``.
        """
        if not request_id:
            logger.error("Cannot verify OTP without request_key from send-OTP step")
            print(" Error: Missing request key from OTP send step")
            return None

        next_request_key = self._verify_otp(request_id, otp)
        if not next_request_key:
            return None

        interim_token = self._verify_pin(next_request_key, pin)
        if not interim_token:
            return None

        return self._get_auth_code(interim_token, fy_id)

    def get_profile_info(self, access_token: str = None) -> Dict[str, Any]:
        """Get user profile information"""
        try:
            token_to_use = access_token or self.access_token
            if not token_to_use:
                return {'error': 'No access token available'}

            headers = {'Authorization': f"{self.client_id}:{token_to_use}"}
            response = requests.get(self.profile_url, headers=headers, timeout=10)

            if response.status_code == 200:
                result = self._parse_json_response(response.text)
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
    """SEBI-compliant daily 2FA CLI-only authentication setup"""
    print("=" * 80)
    print("FYERS API DAILY 2FA AUTHENTICATION SETUP (CLI-ONLY)")
    print("=" * 80)

    try:
        # Check if we already have credentials in environment
        existing_client_id = os.environ.get('FYERS_CLIENT_ID')
        existing_secret_key = os.environ.get('FYERS_SECRET_KEY')

        if existing_client_id and existing_secret_key:
            print("Found existing API credentials in environment")
            print("This will perform FULL re-authentication (not refresh)")

            confirm = input("Proceed with full re-authentication? (y/n) [y]: ").strip().lower()
            if confirm == 'n':
                print("Authentication setup cancelled")
                return False

            auth_manager = FyersAuthManager()
            auth_manager.client_id = existing_client_id
            auth_manager.secret_key = existing_secret_key

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