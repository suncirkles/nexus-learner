"""
scripts/sync_secrets.py
-----------------------
Utility to sync local .env variables to Modal Secrets.
Usage: python scripts/sync_secrets.py
"""

import os
import subprocess
from dotenv import dotenv_values

def sync_secrets():
    print("Reading .env file...")
    config = dotenv_values(".env")
    
    if not config:
        print("Error: .env file not found or empty.")
        return

    # Filter out empty or commented values if necessary
    env_args = []
    for key, value in config.items():
        if value:
            # Escape quotes if needed for the shell
            env_args.append(f"{key}={value}")

    print(f"Syncing {len(env_args)} secrets to Modal...")
    
    # Command: modal secret create nexus-learner-secrets KEY=VAL KEY2=VAL2 ...
    # We use --force to overwrite if it exists
    cmd = ["modal", "secret", "create", "nexus-learner-secrets"] + env_args
    
    try:
        # Note: In some environments, we might need shell=True for Windows
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("Successfully synced secrets to Modal as 'nexus-learner-secrets'.")
        else:
            print(f"Failed to sync secrets: {result.stderr}")
            if "already exists" in result.stderr:
                print("Tip: Use 'modal secret create nexus-learner-secrets --force ...' to overwrite.")
                # Try again with --force
                cmd.insert(3, "--force")
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    print("Successfully overwrote secrets on Modal.")
                else:
                    print(f"Final attempt failed: {result.stderr}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    sync_secrets()
