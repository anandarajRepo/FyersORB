# utils/enhanced_auth_helper.py

"""
Enhanced Fyers Authentication Helper for ORB Trading Strategy
Provides comprehensive authentication with auto-refresh, PIN support, and error handling
"""

import hashlib
import requests
import logging
import os
import json
import getpass
import sys
import time
from typing import Optional, Tuple, Dict, Any

logger = logging.getLogger(__name__)


class FyersAuthManager:
    """Enhanced Fyers authentication manager with refresh token and PIN support"""

    def __init__(self):
        self.client_id = os.environ.get('FYERS_CLIENT_ID')
        self.secret_key = os.environ.get('FYERS_SECRET_KEY')
        self.redirect_uri = os.environ.get('FYERS_REDIRECT_URI', "https://trade.fyers.in/api-login/redirect-to-app")
        self.refresh_token = os.environ.get('FYERS_REFRESH_TOKEN')
        self.access_token = os.environ.get('FYERS_ACCESS_TOKEN')
        self.pin = os.environ.get('FYERS_PIN')

        # API endpoints
        self.auth_url = "https://api-t1.fyers.in/api/v3/generate-authcode"
        self.token_url = "https://api-t1.fyers.in/api/v3/validate-authcode"
        self.refresh_url = "https://api-t1.fyers.in/api/v3/validate-refresh-token"
        self.profile_url = "https://api-t1.fyers.in/api/v3/profile"

    def save_to_env(self, key: str, value: str) -> bool:
        """Save or update environment variable in .env file"""
        try:
            env_file = '.env'

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

    def get_tokens_from_auth_code(self, auth_code: str) -> Tuple[Optional[str], Optional[str]]:
        """Get both access and refresh tokens from auth code"""
        try:
            logger.info("Exchanging auth code for tokens...")

            headers = {"Content-Type": "application/json"}

            data = {
                "grant_type": "authorization_code",
                "appIdHash": self.get_app_id_hash(),
                "code": auth_code
            }

            response = requests.post(self.token_url, headers=headers, json=data, timeout=30)
            response_data = response.json()

            if response.status_code == 200 and response_data.get('s') == 'ok':
                access_token = response_data.get('access_token')
                refresh_token = response_data.get('refresh_token')

                logger.info("Successfully obtained tokens from auth code")
                return access_token, refresh_token
            else:
                error_msg = response_data.get('message', 'Unknown error')
                error_code = response_data.get('code', 'Unknown')
                logger.error(f"Token exchange failed: {error_msg} (Code: {error_code})")
                return None, None

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error during token exchange: {e}")
            return None, None
        except Exception as e:
            logger.error(f"Unexpected error during token exchange: {e}")
            return None, None

    def generate_access_token_with_refresh(self, refresh_token: str) -> Tuple[Optional[str], Optional[str]]:
        """Generate new access token using refresh token with PIN verification"""
        try:
            logger.info("Refreshing access token using refresh token...")

            # Get PIN (from env or user input)
            pin = self.get_or_request_pin()

            headers = {"Content-Type": "application/json"}

            data = {
                "grant_type": "refresh_token",
                "appIdHash": self.get_app_id_hash(),
                "refresh_token": refresh_token,
                "pin": pin
            }

            response = requests.post(self.refresh_url, headers=headers, json=data, timeout=30)
            response_data = response.json()

            if response.status_code == 200 and response_data.get('s') == 'ok':
                new_access_token = response_data.get('access_token')
                new_refresh_token = response_data.get('refresh_token')

                logger.info("Successfully refreshed access token")
                return new_access_token, new_refresh_token
            else:
                error_msg = response_data.get('message', 'Unknown error')
                error_code = response_data.get('code', 'Unknown')

                # Handle specific PIN-related errors
                if 'pin' in error_msg.lower() or 'invalid pin' in error_msg.lower():
                    logger.error(f"PIN verification failed: {error_msg}")
                    print(f"\n PIN verification failed: {error_msg}")
                    print("The saved PIN might be incorrect.")

                    # Clear the saved PIN and retry once
                    self.pin = None
                    if 'FYERS_PIN' in os.environ:
                        del os.environ['FYERS_PIN']

                    retry = input("Would you like to retry with a different PIN? (y/n): ").strip().lower()
                    if retry == 'y':
                        logger.info("Retrying token refresh with new PIN...")
                        return self.generate_access_token_with_refresh(refresh_token)
                else:
                    logger.error(f"Token refresh failed: {error_msg} (Code: {error_code})")

                return None, None

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error while refreshing token: {e}")
            return None, None
        except Exception as e:
            logger.error(f"Unexpected error while refreshing token: {e}")
            return None, None

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
        """Get a valid access token, using refresh token if available"""
        try:
            # First, check if current access token is still valid
            if self.access_token and self.is_token_valid(self.access_token):
                logger.info("Current access token is still valid")
                return self.access_token

            logger.info("Access token is invalid or expired")

            # Try to use refresh token if available
            if self.refresh_token:
                logger.info("Attempting to refresh access token...")
                new_access_token, new_refresh_token = self.generate_access_token_with_refresh(self.refresh_token)

                if new_access_token:
                    logger.info("Successfully refreshed access token")

                    # Save new tokens
                    self.save_to_env('FYERS_ACCESS_TOKEN', new_access_token)
                    self.access_token = new_access_token

                    if new_refresh_token:
                        self.save_to_env('FYERS_REFRESH_TOKEN', new_refresh_token)
                        self.refresh_token = new_refresh_token

                    return new_access_token
                else:
                    logger.warning("Failed to refresh access token")

            # If refresh failed or no refresh token, require full authentication
            logger.info("Full re-authentication required")
            return self.setup_full_authentication()

        except Exception as e:
            logger.error(f"Error getting valid access token: {e}")
            return None

    def setup_full_authentication(self) -> Optional[str]:
        """Complete authentication flow to get new tokens"""
        try:
            print("\n" + "=" * 70)
            print("FYERS API FULL AUTHENTICATION SETUP")
            print("=" * 70)

            if not all([self.client_id, self.secret_key]):
                print(" Missing CLIENT_ID or SECRET_KEY in environment variables")
                return None

            # PIN setup reminder
            if not self.pin:
                print("\nTrading PIN Setup:")
                print("Your trading PIN will be needed for future token refreshes.")
                print("We'll ask for it during the authentication process.")

            # Generate auth URL
            auth_url = self.generate_auth_url()
            if not auth_url:
                print(" Failed to generate authentication URL")
                return None

            print(f"\n AUTHENTICATION STEPS:")
            print(f"1. Open this URL in your web browser:")
            print(f"   {auth_url}")
            print(f"\n2. Log in to your Fyers account")
            print(f"3. Complete the authorization process")
            print(f"4. Copy the authorization code from the redirect URL")
            print(f"\n The redirect URL will look like:")
            print(f"   {self.redirect_uri}?code=YOUR_AUTH_CODE&state=sample_state")

            # Get auth code from user
            print(f"\n" + "=" * 50)
            auth_code = input(" Enter the authorization code: ").strip()

            if not auth_code:
                print(" No authorization code provided")
                return None

            # Get both access and refresh tokens
            print(" Exchanging authorization code for tokens...")
            access_token, refresh_token = self.get_tokens_from_auth_code(auth_code)

            if not access_token:
                print(" Failed to obtain access token")
                return None

            print(" Tokens obtained successfully!")

            # Set up PIN for future token refreshes
            try:
                print("\n Setting up PIN for automatic token refresh...")
                pin = self.get_or_request_pin()
                if pin:
                    print(" PIN configured for automatic token refresh")
            except Exception as e:
                print(f"  PIN setup skipped: {e}")
                print("You can set it up later with: python main.py update-pin")

            # Save all tokens to .env
            print(f"\n Saving authentication data...")

            saved_items = []
            if self.save_to_env('FYERS_ACCESS_TOKEN', access_token):
                saved_items.append("Access Token")

            if refresh_token and self.save_to_env('FYERS_REFRESH_TOKEN', refresh_token):
                saved_items.append("Refresh Token")

            if saved_items:
                print(f" Saved: {', '.join(saved_items)}")

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
    """Enhanced authentication setup with refresh token and PIN support"""
    print("=" * 80)
    print("ENHANCED FYERS API AUTHENTICATION SETUP")
    print("=" * 80)

    try:
        # Check if we already have credentials in environment
        existing_client_id = os.environ.get('FYERS_CLIENT_ID')
        existing_secret_key = os.environ.get('FYERS_SECRET_KEY')

        if existing_client_id and existing_secret_key:
            print("Found existing API credentials in environment")
            print("This will perform FULL re-authentication (not refresh)")

            confirm = input("\nProceed with full re-authentication? (y/n) [y]: ").strip().lower()
            if confirm == 'n':
                print("Authentication setup cancelled")
                return False

            auth_manager = FyersAuthManager()
            auth_manager.client_id = existing_client_id
            auth_manager.secret_key = existing_secret_key

            # Skip refresh attempt - go directly to full authentication
            print("\nStarting full authentication flow...")
            access_token = auth_manager.setup_full_authentication()

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

        # Perform full authentication
        access_token = auth_manager.setup_full_authentication()

        if access_token:
            print("\nEnhanced authentication setup completed!")
            print("✓ Refresh token and PIN have been configured for automatic renewal")
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
    """Handle Fyers authentication with refresh token and PIN support"""
    try:
        auth_manager = FyersAuthManager()

        # Get valid access token (will auto-refresh if needed)
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
            print("The new PIN will be used for automatic token refresh")
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
        print(f"Refresh Token configured: {'Yes' if auth_manager.refresh_token else 'No'}")
        print(f"PIN configured: {'Yes' if auth_manager.pin else 'No'}")

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