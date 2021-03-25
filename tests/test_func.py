import pyarrow as pa 
from wombat_db import join, filters, groupby, head, drop_duplicates

# Create data
t = pa.Table.from_pydict({
    'Animal': ['Falcon', 'Falcon', 'Parrot', 'Parrot', 'Parrot'],
    'Max Speed': [380., 370., 24., 26., 24.]
})
print("Source:")
head(t)

# Drop duplicates
print("Drop duplicates:")
d = drop_duplicates(t, on=['Animal'], keep='first')
head(d)

# Groupby aggregations
print("Groupby loop:")
for key, value in groupby(t, ['Animal']):
    print(key)
    head(value)

print("Aggregrations:")
g = groupby(t, ['Animal']).agg({'Max Speed': 'max', 'Max Speed Avg.': ('Max Speed', 'mean')})
head(g)

# Filters
print("Filters:")
f = filters(t, ('Animal', '=', 'Falcon'))
f = filters(t, [('Animal', 'not in', ['Falcon', 'Duck']), ('Max Speed', '<', 25)])
head(f)

# Join operations
print("Join:")
t2 = pa.Table.from_pydict({
    'Animal': ['Falcon', 'Parrot'],
    'Age': [10, 20]
})
j = join(t, t2, on=['Animal'])
head(j)