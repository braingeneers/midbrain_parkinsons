# For writing commands that will be executed after the container is created

# Installs `human_hip` as local library without resolving dependencies (--no-deps)
#python3 -m pip install -e /workspaces/human_hippocampus --no-deps
#python3 -m pip install -e /workspaces/human_hippocampus
python3 -m pip install diptest
python3 -m pip install PyWavelets
python3 -m pip install spkit
python3 -m pip install astropy
python3 -m pip install statsmodels

# # Loop through the dataset names to download them for S3
# for data in "2023-04-02-e-hc328_unperturbed/derived/"  "2022-11-02-e-Hc11.1-chip16753/derived/" "2023-05-10-e-hc52_18790_unperturbed/derived/" "2023-05-10-e-hc52_18790/derived/" 
# # note- we exclude a potential data, "2022-10-20-e-/derived/", infant dentate, feel free to download and analyze it! 
# do
#   # Create directories and download data from aws into them
#   mkdir -p "/workspaces/human_hippocampus/data/ephys/$data"
#   aws --endpoint https://s3.braingeneers.gi.ucsc.edu s3 cp "s3://braingeneers/ephys/$data" "/workspaces/human_hippocampus/data/ephys/$data" --recursive --no-sign-request
# done

# change folder name for downlaoded data? (might not be necessary)
#mv /workspaces/human_hippocampus/data/ephys/2023-11-13-e-Hc110723_hckcr1_21841/derived2 /workspaces/human_hippocampus/data/ephys/2023-11-13-e-Hc110723_hckcr1_21841/derived


# Create curated versions of the data
# python3 /workspaces/human_hippocampus/.devcontainer/curate_data.py
