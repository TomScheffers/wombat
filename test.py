from wombat.executor import Database
from pyarrow_ops import head
import pyarrow.parquet as pq
import time

# Avoid initial take() time
t1 = pq.ParquetDataset('data/skus/org_key=0/file0.parquet').read(columns=['sku_key'])
t1.take([1, 2, 3])

# Read data
time0 = time.time()
d1 = pq.ParquetDataset('data/skus')
d2 = pq.ParquetDataset('data/stock_current') #.read(columns=['sku_key', 'store_key', 'economical', 'technical'])

# Database
db = Database(cache=True)
db.register_dataset('skus', d1)
db.register_dataset('stock_current', d2)

# 1. Create query plan + collect it
time1 = time.time()
p = db['skus'] \
    .join('stock_current', on=['org_key', 'sku_key']) \
    .filter([('org_key', '=', 0), ('store_key', '<=', 200)]) \
    .groupby(by=['option_key']) \
    .agg({'economical': 'sum'}) \
    .orderby('economical', ascending=False)

# Collect first optimizes the plan by doing: 1. backward filter propagation and 2. backward column propagation (avoid reading useless columns)
r0 = p.collect()
head(r0)

time2 = time.time()
# 2. Same execution plan structured differently
r = db['stock_current'].filter([('org_key', '=', 0), ('store_key', '<=', 200)])
l = db['skus']
g = l.join(r, on=['org_key', 'sku_key']) \
    .groupby(by=['option_key']) \
    .agg({'economical': 'mean'}) \
    .orderby('economical', ascending=True)

# Some notes:
# Skus is filtered by org_key=0 because of the inner join operator
# In the join, the org_key is skipped, because it only has 1 value on both sides
# The groupby operation hits the caching layer as the underlying structure is the same 

r1 = g.collect()
head(r1)
print(time1 - time0, time2 - time1, time.time() - time2)