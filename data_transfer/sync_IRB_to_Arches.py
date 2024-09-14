import os
import time
import subprocess
from threading import Timer

# Define your sync command
mount_command = r"net use R: \\R01PUGHSM03.R01.MED.VA.GOV\Research"
sync_command = r"aws s3 sync R:\\Kicska_Gregory\\1704083\\1704083_Arches\\clearcanvas s3://project-sic-s3/data/raw/VA/clearcanvas_testset"
reauth_command = r"aws-adfs login --adfs-host=prod.adfs.federation.va.gov --provider-id urn:amazon:webservices:govcloud --region us-gov-west-1"


def schedule_reauth():
    subprocess.run(reauth_command, shell=True)
    subprocess.run(mount_command, shell=True)
    print("Re-authenticated successfully.")
    # Sync immediately after reauth
    run_sync()
    Timer(900, schedule_reauth).start()
    # 900 seconds = 15 minutes


def run_sync():

    # Start the first re-authentication
    while True:
        try:
            # Run the sync command print("Starting sync operation...")
            subprocess.run(sync_command, shell=True, check=True)
            print("Sync operation completed successfully.")
            break
        # Exit loop if sync is successful
        except subprocess.CalledProcessError as e:
            print(f"Sync operation failed: {e}")
            print("Retrying sync after a short delay...")
            time.sleep(10)


schedule_reauth()
run_sync()
