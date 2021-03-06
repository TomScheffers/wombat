from wombat.executor import Database
from pyarrow_ops import head
import pyarrow.parquet as pq
import time

# Read data
time0 = time.time()
d1 = pq.ParquetDataset('data/skus')
d2 = pq.ParquetDataset('data/stock_current') #.read(columns=['sku_key', 'store_key', 'economical', 'technical'])

# Database
db = Database(cache=True)
db.register_dataset('skus', d1)
db.register_dataset('stock_current', d2)

# 1. Create query plan + collect it
df = db['stock_current']
df['ttl'] = df['economical'] + df['technical'] + df['economical']
# df['econ+2'] = df['economical'] + 2


r = df.collect()
head(r)