# Install braingeneers
${NB_PYTHON_PREFIX}/bin/pip install --no-cache git+https://github.com/uw-ssec/braingeneerspy.git#egg=braingeneers[data,iot,analysis,ml]



# Define the base directory path
base_dir="/workspaces/human_hippocampus/data/ephys"

# Define an array of dataset names
dataset_names=("2023-04-02-e-hc328_unperturbed/derived/" "2022-10-20-e-/derived/" "2022-11-02-e-Hc11.1-chip16753/derived")

# Loop through the dataset names
for sub_dir in "${dataset_names[@]}"; do
  # Create directories
  mkdir -p "$base_dir/$sub_dir"

  # Download data from S3
  aws --endpoint https://s3-west.nrp-nautilus.io s3 cp "s3://braingeneers/ephys/$sub_dir" "$base_dir/$sub_dir" --recursive --no-sign-request
done






#. Data folders:
