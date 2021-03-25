# Wombat
Wombat is Python libary for data crunching operations directly on the pyarrow.Table class, implemented in numpy & Cython. For convenience, function naming and behavior tries to replicates that of the Pandas API / Postgresql language.

Current features:
- Engine API (lazy execution):
    - Operate directly on Pyarrow tables and datasets
    - Filter push-downs to optimize speed (only read subset of partitions)
    - Column tracking: only read subset of columns in data
    - Many operations (join, aggregate, filters, drop_duplicates, ...)
    - Numerical / logical operations on Column references
    - Caching based on hashed subtrees and reference counting
    - Visualize Plan using df.plot(file) (required graphviz)
- Operation API (direct execution): 
    - Data operations like joins, aggregations, filters & drop_duplicates
- ML preprocessing API: 
    - Categorical, numericals and one-hot processing directly on pa.Tables
    - Reusable: Serialize cleaners to JSON for using in inference
- SQL API (under construction)
- DB Management API (under construction)

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install wombat.

```bash
pip install wombat_db
```

## Usage
See tests folder for more code examples

Dataframe API:
```python
from wombat import Engine, head
import pyarrow.parquet as pq

# Create Engine and register_dataset/table
db = Engine(cache_memory=1e9)
db.register_dataset('skus', pq.ParquetDataset('data/skus'))
db.register_dataset('stock_current', pq.ParquetDataset('data/stock_current'))

# Selecting a table from db generates a Plan object
df = db['stock_current']

# Operations can be chained, adding nodes to the Plan
df = df.filter([('org_key', '=', 0), ('store_key', '<=', 200)]) \
    .join(db['skus'], on=['org_key', 'sku_key']) \
    .aggregate(by=['option_key'], methods={'economical': 'sum', 'technical':'max'})

# Selecting strings from the Dataframe object, yields a column reference
df['stock'] = df['economical'].coalesce(0).least(df['technical']).greatest(0)

# A column reference can be used for numerical & logical operations
df['calculated'] = ((df['stock'] - 100) ** 2 / 5000 - df['stock']).clip(None, 5000)
df['check'] = ~(df['calculated'] == 5000) and (df['stock'] > 10000)

# We can filter using the boolean column as value
df[~(df['calculated'] == 5000)]

# Register UDF (pa.array -> pa.array)
db.register_udf('power', lambda arr: pa.array(arr.to_numpy() ** 2))
df['economical ** 2'] = df.udf('power', df['economical'])

# Rename columns
df.rename({'economical': 'economical_sum', 'technical': 'technical_max'})

# Select a subselection of columns (not necessary)
df.select(['option_key', 'economical_sum', 'calculated', 'check', 'economical ** 2'])

# You do not need to catch the return for chaining of operations
df.orderby('calculated', ascending=False)

# Collect is used to execute the plan
r = df.collect(verbose=True)
head(r)

# Cache is hit when same operations are repeated
# JOIN hits cache here, as filters are propagated down
df = db['stock_current'] \
    .join(db['skus'], on=['org_key', 'sku_key']) \
    .filter([('org_key', '=', 0), ('store_key', '<=', 200)]) \
    .aggregate(by=['option_key'], methods={'economical': 'max', 'technical':'sum'}) \
    .orderby('economical', ascending=False)
r = df.collect(verbose=True)
head(r)
```

### To Do's
- [ ] Add unit tests using pytest
- [ ] Add more join options (left, right, outer, full, cross)
- [ ] Track schema in forward pass
- [ ] Improve groupify operation for multi columns joins / groups
- [ ] Serialize cache (to disk)
- [ ] Serialize database (to disk)

## Contributing
Pull requests are very welcome, however I believe in 80% of the utility in 20% of the code. I personally get lost reading the tranches of complicated code bases. If you would like to seriously improve this work, please let me know!