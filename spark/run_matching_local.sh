# cluster ds-ob-dev02
DB_NAME="guardian"

# 4 for BB authored news, 6 for web-scraped content
#for PVF_LEVEL in 4 6; do
python3 runner.py \
      --db_name $DB_NAME \
      --num_files 500 \
      --continuous \
      --env pluslab
