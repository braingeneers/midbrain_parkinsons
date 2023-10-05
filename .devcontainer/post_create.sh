# For writing commands that will be executed after the container is created

# Installs `human_hip` as local library without resolving dependencies (--no-deps)
python3 -m pip install -e /workspaces/human_hippocampus --no-deps
python3 -m pip install diptest


# Install datasets from S3
# This script will download the datasets from S3 and place them in the correct directory structure
# This script should be run from the root of the repository

# Define and create base directory path
base_dir="/workspaces/human_hippocampus/data/ephys"
mkdir -p "$base_dir"
# Download all the spikesorted datasets by name
dataset_names=("2023-04-02-e-hc328_unperturbed/derived/" "2022-10-20-e-/derived/" "2022-11-02-e-Hc11.1-chip16753/derived" "2023-05-10-e-hc52_18790_unperturbed/derived")
# Loop through the dataset names
for sub_dir in "${dataset_names[@]}"; do
  # Create directories
  mkdir -p "$base_dir/$sub_dir"
  # Download data from S3
  aws --endpoint https://s3-west.nrp-nautilus.io s3 cp "s3://braingeneers/ephys/$sub_dir" "$base_dir/$sub_dir" --recursive --no-sign-request
done

# Create curated versions of the data
python3 /workspaces/human_hippocampus/src/curate_data.py
