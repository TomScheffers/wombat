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
df = db['overview'].filter(('org_key', '=', 2))
df.fillna(['stock', 'stock_extra', 'potential_n5w', 'stock_goal', 'projection_n5w'], 0)

# Acces struct element
df['dimension'] = df['properties']['BK_Article']

# Aggregate
df.aggregate(
    by='dimension', 
    methods={
        'stock': 'sum',
        'stock_extra': 'sum',
        'potential_n5w': 'sum',
        'stock_goal': 'sum',
        'projection_n5w': 'sum',
    }
)
df.filter(('stock_extra', '>', 100))

df.orderby('stock', ascending=False)

# Collect
t = df.collect(verbose=True)
head(t)