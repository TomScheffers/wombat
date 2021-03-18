from wombat_db import Engine, head
import pyarrow.parquet as pq

# Avoid initial take() time
t1 = pq.ParquetDataset('data/skus/org_key=0/file0.parquet').read(columns=['sku_key'])
t1.take([1, 2, 3])

# Read data
d1 = pq.ParquetDataset('data/skus')
d2 = pq.ParquetDataset('data/stock_current')

# Database and register tables
db = Engine(cache=True)
db.register_dataset('skus', d1)
db.register_dataset('stock_current', d2)

df = db.sql("SELECT option_key, SUM(economical) as economical FROM stock_current JOIN skus USING (org_key, sku_key) WHERE org_key = 0 AND store_key <= 200 GROUP BY option_key ORDER BY economical DESC")
r = df.collect()
head(r)

# Plot
df.plot('graph')