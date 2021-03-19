from wombat_db import Engine, head
import pyarrow.parquet as pq

# Avoid initial take() time
t1 = pq.ParquetDataset('data/skus/org_key=0/file0.parquet').read(columns=['sku_key'])
t1.take([1, 2, 3])

# Read data
d1 = pq.ParquetDataset('data/overview', validate_schema=False)

# Database and register tables
db = Engine(cache_memory=1e9)
db.register_dataset('overview', d1)

# Look at properties
df = db['overview']
df.filter(('org_key', '=', 2))
df.fillna(['stock', 'stock_extra', 'potential_n5w', 'stock_goal', 'projection_n5w'], 0)

# Acces struct element
df['dimension'] = df['properties']['ColorCode']

# Aggregate
df.aggregate(
    by='dimension', 
    methods={
        'stock': 'sum',
        'stock_extra': 'sum',
        'potential_n5w': 'sum',
        'stock_goal': 'sum',
        'projection_n5w': 'sum',
        'projection_n5w_max': ('projection_n5w', 'max'),
        'skus': ('stock', 'count'),
        'options': ('option_key', 'distinct_count'),
    }
)
df.cast({'projection_n5w': int, 'potential_n5w': int})
df.filter(('stock_extra', '>', 100))

df.orderby('stock', ascending=False)

# Collect
t = df.collect(verbose=True)
head(t)