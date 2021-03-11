from wombat import Database
from pyarrow_ops import head
import pyarrow.parquet as pq

# Avoid initial take() time
t1 = pq.ParquetDataset('data/skus/org_key=0/file0.parquet').read(columns=['sku_key'])
t1.take([1, 2, 3])

# Read data
d1 = pq.ParquetDataset('data/skus')
d2 = pq.ParquetDataset('data/stock_current')

# Database and register tables
db = Database(cache=True)
db.register_dataset('skus', d1)
db.register_dataset('stock_current', d2)

# Selecting from db generates a Dataframe object
df = db['stock_current']

# Operations can be chained, adding nodes to the Plan
df = df.filter([('org_key', '=', 0), ('store_key', '<=', 200)]) \
    .join(db['skus'], on=['org_key', 'sku_key']) \
    .aggregate(
        by=['option_key'],
        methods={
            'economical': 'sum', 
            'technical':'max'
        }
    )

# Selecting strings from the Dataframe object, yields a column reference
df['stock'] = df['economical']

# A column reference can be used for numerical & logical operations
df['calculated'] = (df['stock'] - 100) ** 2 / 5000 - df['stock']
df['threshold'] = ~(df['calculated'] > 5000)

# We can filter using the boolean column as value
df[(df['stock'] < 20000)]

# TODO: make df[df['threshold']] work

# Rename columns
df.rename({
    'economical': 'economical_sum',
    'technical': 'technical_max'
})

# Select columns
df.select(['option_key', 'economical_sum', 'calculated', 'threshold'])

# You do not need to catch the return for chaining of operations
df.orderby('calculated', ascending=False)

# Collect is used to execute the plan
r = df.collect()
head(r)

# Cache is hit when same operations are repeated
# JOIN hits cache here, as filters are propagated down
df = db['stock_current'] \
    .join(db['skus'], on=['org_key', 'sku_key']) \
    .filter([('org_key', '=', 0), ('store_key', '<=', 200)]) \
    .aggregate(
        by=['option_key'],
        methods={
            'economical': 'max', 
            'technical':'sum'
        }
    ) \
    .orderby('economical', ascending=False)
r = df.collect()
head(r)



