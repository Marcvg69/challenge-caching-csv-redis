python main.py --clear-cache
python app/main.py --query avg_dep_delay_by_month
python app/main.py --query avg_arr_delay_by_airline   
python app/main.py --query flights_by_origin
python main.py --show-cache

# Use your existing compose Redis config (host 6380)
export REDIS_HOST=localhost
export REDIS_PORT=6380

# .env will be auto-loaded; CSV_PATH will point to your big flights file
python app/main.py --clear-cache
python app/main.py --query avg_arr_delay_by_airline --limit=5   # MISS -> caches
python app/main.py --query avg_arr_delay_by_airline --limit=5   # HIT

python app/main.py --query flights_by_origin --limit=5
python app/main.py --query avg_dep_delay_by_month --limit=5

python app/main.py --show-cache
python app/main.py --show-cache --full   # pretty JSON
