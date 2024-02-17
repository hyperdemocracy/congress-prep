import duckdb
import pandas as pd
import requests

con = duckdb.connect()
con.execute("INSTALL httpfs;")
con.execute("LOAD httpfs;")

ds_name = "hyperdemocracy/us-congress"

response = requests.get(f"https://datasets-server.huggingface.co/parquet?dataset={ds_name}")
urls = [f['url'] for f in response.json()['parquet_files'] if f['config'] == 'unified_v1']
print(urls)

sys.exit(0)

# you can read the parquet files remotely into a dataframe if you want
df = pd.concat([pd.read_parquet(url) for url in urls]).reset_index(drop=True)

# or you can query them remotely using duckdb and SQL syntax
query = f"SELECT * FROM read_parquet({urls}) WHERE legis_id = '118-hconres-1'"
res = con.sql(query)
dfq = res.fetchdf()
