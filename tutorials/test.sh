
# Loop through the dataset names
for data in "2023-04-02-e-hc328_unperturbed/derived/"  "2022-11-02-e-Hc11.1-chip16753/derived/" "2023-05-10-e-hc52_18790_unperturbed/derived/"
# note- we exclude a potential data, "2022-10-20-e-/derived/", infant dentate, feel free to download and analyze it! 
do
  # Create directories and download data from aws into them
  mkdir -p "/workspaces/human_hippocampus/data/ephys/$data"
  aws --endpoint https://s3-west.nrp-nautilus.io s3 cp "s3://braingeneers/ephys/$data" "/workspaces/human_hippocampus/data/ephys/$data" --recursive --no-sign-request
done



