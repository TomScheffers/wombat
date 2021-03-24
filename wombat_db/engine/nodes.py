import pyarrow as pa
import numpy as np
from wombat_db.ops import join, groupby, filters
from wombat_db.engine.column import ColumnNode
import hashlib, json, time

# Computation nodes
class BaseNode():
    def check(self, needed, reference):
        missing = [c for c in needed if c not in reference]
        if missing:
            raise Exception("Required columns ({}) not in available columns ({})".format(", ".join(missing), ", ".join(reference)))

    def columns_bw(self, columns_backward):
        self.columns_backward = [c for c in sorted(list(set(self.columns_forward + columns_backward))) if c in self.columns_source]
        return self.columns_backward

    def properties(self):
        fields = ['table', 'on', 'filters', 'by', 'methods', 'key', 'ascending', 'calculation', 'columns_backward']
        obj = {k: v for k,v in self.__dict__.items() if k in fields}
        return {**{'name': self.__class__.__name__}, **obj}

    def graph_info(self):
        obj = self.properties()
        return "\\l".join([k + ': ' + str(v).replace("'", "") for k,v in obj.items() if k not in ['columns_backward']])

    def hash(self, h=None):
        # We want to add: on, by, columns_backward, table_url
        if not h:
            h = hashlib.sha256()
        h.update(json.dumps(self.properties(), sort_keys=True).encode())
        self.hash_key = h.hexdigest()
        return h

    def backward(self, columns_backward=[], filters_backward=[]):
        self.columns_bw(columns_backward)
        hp = self.parent.backward(columns_backward=self.columns_backward, filters_backward=filters_backward)
        return self.hash(h=hp)

    def get(self, verbose):
        self.time = time.time()
        if self.cache and self.hash_key in self.cache_obj.keys():
            t = self.cache_obj[self.hash_key]
            if verbose:
                print("Node: {} Rows: {} Cumulative Time: {:2f} (cached)".format(self.__class__.__name__.ljust(16), str(t.num_rows).ljust(9), time.time() - self.time))
        else:
            t = self.fetch(verbose)
            if verbose:
                print("Node: {} Rows: {} Cumulative Time: {:2f}".format(self.__class__.__name__.ljust(16), str(t.num_rows).ljust(9), time.time() - self.time))
            if self.cache:
                self.cache_obj.put(self.hash_key, t, weight=time.time() - self.time)
        return t

# Sources
class TableNode(BaseNode):
    def __init__(self, table, database, cache_obj=None):
        self.table, self.database, self.cache_obj = table, database, cache_obj
        self.cache = (cache_obj != None)
        self.table = self.database.tables[table]
        self.columns = self.table.column_names
        self.columns += list(set([c.split('.')[0] for c in self.columns if '.' in c]))

        # Forward propagation of nodes
        self.columns_source, self.columns_forward, self.filters_forward = self.columns, [], []

    def backward(self, columns_backward=[], filters_backward=[]):
        self.columns_bw(columns_backward)
        self.filters = self.filters_backward
        return self.hash()

    def fetch(self, verbose):
        t = self.table.select(self.columns_backward)
        tf = (filters(t, self.filters) if self.filters else t)
        return tf

def part_check(part, op, value):
    # Try to cast partition to value
    try:
        part = type(value)(part)
    except:
        raise Exception("Cannot downcast {} to data type {}".format(part, type(value)))

    if op in ['=', '==']:
        return part == value
    elif op == '!=':
        return part != value
    elif op == '<':
        return part < value
    elif op == '>':
        return part > value
    elif op == '<=':
        return part <= value
    elif op == '>=':
        return part >= value
    elif op == 'in':
        return part in value
    elif op == 'not in':
        return part not in value
    else:
        raise Exception("Operand {} is not implemented!".format(op))

class DatasetNode(BaseNode):
    def __init__(self, table, database, cache_obj=None):
        self.table, self.database, self.cache_obj = table, database, cache_obj
        self.cache = (cache_obj != None)
        self.dataset = database.datasets[table]

        self.partition_keys = [p.name for p in self.dataset.partitions]
        self.partition_values = [{pk[0]: dp.keys[pk[1]] for pk, dp in zip(p.partition_keys, self.dataset.partitions)} for p in self.dataset.pieces]
        self.meta = self.dataset.pieces[0].get_metadata()
        self.columns = self.partition_keys + [c['path_in_schema'] for c in self.meta.row_group(0).to_dict()['columns']]
        self.columns += list(set([c.split('.')[0] for c in self.columns if '.' in c]))

        # Forward propagation of nodes
        self.columns_source, self.columns_forward, self.filters_forward = self.columns, [], []

    def backward(self, columns_backward=[], filters_backward=[]):
        self.columns_bw(columns_backward)
        self.filters = filters_backward
        self.part_filters = list(filter(lambda f: f[0] in self.partition_keys, self.filters))
        self.value_filters = list(filter(lambda f: f[0] not in self.partition_keys, self.filters))
        return self.hash()

    def partition_check(self, partition_value, filters):
        for key, op, value in filters:
            if not part_check(partition_value[key], op, value):
                return False
        return True

    def fetch(self, verbose):
        ts = []
        for i, p in enumerate(self.dataset.pieces):
            if self.partition_check(self.partition_values[i], self.part_filters):
                ts.append(p.read(columns=[c for c in self.columns_backward if c not in self.partition_keys], partitions=self.dataset.partitions))
        t = pa.concat_tables(ts)
        return (filters(t, self.value_filters) if self.value_filters else t)

# Operations
def column_min_max(arr):
    if hasattr(arr, 'dictionary'):
        mmx = pa.compute.min_max(arr.indices)
        return (arr.dictionary[mmx['min'].as_py()].as_py(), arr.dictionary[mmx['max'].as_py()].as_py())
    else:
        mmx = pa.compute.min_max(arr)
        return (mmx['min'].as_py(), mmx['max'].as_py())

class JoinNode(BaseNode):
    def __init__(self, left, right, on, cache_obj=None):
        self.left, self.right, self.on, self.cache_obj = left, right, (on if isinstance(on, list) else [on]), cache_obj
        self.cache = (cache_obj != None)

        # Check columns
        self.check(needed=self.on, reference=left.columns)
        self.check(needed=self.on, reference=right.columns)
        self.columns = list(set(left.columns + right.columns))

        # Forward propagation of nodes
        self.columns_source, self.columns_forward = list(set(left.columns_source + right.columns_source)), list(set(left.columns_forward + right.columns_forward + on))
        self.filters_forward = left.filters_forward + right.filters_forward #[f for f in left.filters_forward + right.filters_forward if f[0] in self.on]

    def backward(self, columns_backward=[], filters_backward=[]):
        self.columns_bw(columns_backward)
        self.filters = filters_backward
        filters_l, filters_r = [f for f in self.filters if f[0] in self.left.columns_source], [f for f in self.filters if f[0] in self.right.columns_source]
        columns_l, columns_r = [c for c in self.columns_backward if c in self.left.columns_source], [c for c in self.columns_backward if c in self.right.columns_source]
        hl = self.left.backward(columns_backward=columns_l, filters_backward=filters_l)
        hr = self.right.backward(columns_backward=columns_r, filters_backward=filters_r)
        hl.update(hr.digest())
        return self.hash(h=hl)

    def fetch(self, verbose):
        tl = self.left.get(verbose)
        tr = self.right.get(verbose)
        on = []
        for i, o in enumerate(self.on):
            l_mmx, r_mmx = column_min_max(tl.column(o).combine_chunks()), column_min_max(tr.column(o).combine_chunks())

            # Skip if there is only 1 value in both tables
            if (l_mmx == r_mmx) and (l_mmx[0] == l_mmx[1]):
                continue
            
            # Add column to join condition
            on.append(o)

            # Filter min max values, before joining
            # tlf = filters(tl, [(o, '>=', max(l_mmx[0], r_mmx[0])), (o, '<=', min(l_mmx[1], r_mmx[1]))])
            # trf = filters(tr, [(o, '>=', max(l_mmx[0], r_mmx[0])), (o, '<=', min(l_mmx[1], r_mmx[1]))])
        return join(left=tl, right=tr, on=on)

class FilterNode(BaseNode):
    def __init__(self, parent, filters, cache_obj=None):
        self.parent, self.filters, self.cache_obj = parent, ([filters] if isinstance(filters, tuple) else filters), cache_obj
        self.cache = (cache_obj != None)

        # Check if columns are available
        self.check(needed=[f[0] for f in self.filters], reference=parent.columns)
        self.columns = parent.columns

        # Forward propagation of nodes
        self.columns_source, self.columns_forward = parent.columns_source, list(set(parent.columns_forward + [f[0] for f in self.filters if f[0] in parent.columns_source]))
        self.filters_forward = parent.filters_forward + self.filters

    def backward(self, columns_backward=[], filters_backward=[]):
        self.columns_bw(columns_backward)
        hp = self.parent.backward(columns_backward=self.columns_backward, filters_backward=filters_backward)
        self.hash_key = hp.hexdigest() # Filter node does not change anything, so can just pass its parents hash
        return hp

    def fetch(self, verbose):
        return self.parent.get(verbose)

class AggregateNode(BaseNode):
    def __init__(self, parent, by, methods, cache_obj=None):
        self.parent, self.by, self.methods, self.filters, self.cache_obj = parent, (by if isinstance(by, list) else [by]), methods, [], cache_obj
        self.cache = (cache_obj != None)

        # Check if columns are available
        refs = [(m[0] if isinstance(m, tuple) else k) for k, m in methods.items()]
        self.check(needed=self.by + refs, reference=parent.columns)
        self.columns = self.by + list(methods.keys())

        # Forward propagation of nodes
        self.columns_source, self.columns_forward, self.filters_forward = parent.columns_source, list(set(parent.columns_forward + [c for c in self.by + refs if c in parent.columns_source])), parent.filters_forward

    def backward(self, columns_backward=[], filters_backward=[]):
        self.columns_bw(columns_backward)
        self.filters = [f for f in filters_backward if f[0] in self.methods.keys()] # Intercept filters which are aggregate values
        hp = self.parent.backward(columns_backward=self.columns_backward, filters_backward=[f for f in filters_backward if f not in self.filters])
        return self.hash(h=hp)

    def fetch(self, verbose):
        tp = self.parent.get(verbose)
        t = groupby(tp, self.by).agg(self.methods)
        return (filters(t, self.filters) if self.filters else t)

class OrderNode(BaseNode):
    def __init__(self, parent, key, ascending, cache_obj=None):
        self.parent, self.key, self.ascending, self.cache_obj = parent, key, ascending, cache_obj
        self.cache = (cache_obj != None)

        # Check if columns are available
        self.check(needed=[key], reference=parent.columns)
        self.columns = parent.columns 
    
        # Forward propagation of nodes
        self.columns_source, self.columns_forward, self.filters_forward = parent.columns_source, list(set(parent.columns_forward + ([self.key] if self.key in parent.columns_source else []))), parent.filters_forward

    def fetch(self, verbose):
        tp = self.parent.get(verbose)
        idxs = pa.compute.sort_indices(tp.column(self.key)).to_numpy()
        return (tp.take(idxs) if self.ascending else tp.take(idxs[::-1]))

class SelectionNode(BaseNode):
    def __init__(self, parent, columns=[], aliases=[], cache_obj=None):
        self.parent, self.columns, self.aliases, self.cache_obj = parent, columns, aliases, cache_obj
        if not columns:
            self.columns = [c for c in self.parent.columns if '.' not in c]
        self.cache = (cache_obj != None)
        self.mapping = (dict(zip(columns, aliases)) if aliases else {})

        # Check if columns are available
        self.check(needed=columns, reference=parent.columns)
        if aliases: 
            self.columns = [self.mapping.get(c, c) for c in parent.columns]

        # Forward propagation of nodes
        self.columns_source, self.columns_forward, self.filters_forward = parent.columns_source, list(set(parent.columns_forward + [c for c in columns if c in parent.columns_source])), parent.filters_forward
        
    def fetch(self, verbose):
        tp = self.parent.get(verbose)
        if self.aliases:
            return tp.rename_columns([self.mapping.get(col, col) for col in tp.column_names])
        else:
            return tp.select(self.columns)

class CalculationNode(BaseNode):
    def __init__(self, parent, key, column, cache_obj=None):
        self.parent, self.key, self.calculation, self.column, self.cache_obj = parent, key, column.key, column, cache_obj
        self.cache = (cache_obj != None)

        # Check if columns are available
        self.check(needed=column.required, reference=parent.columns)
        self.columns = parent.columns + [key]
    
        # Forward propagation of nodes
        self.columns_source, self.columns_forward, self.filters_forward = parent.columns_source, list(set(parent.columns_forward + [c for c in column.required if c in parent.columns_source])), parent.filters_forward
    
    def backward(self, columns_backward=[], filters_backward=[]):
        self.columns_bw(columns_backward)
        self.filters = [f for f in filters_backward if f[0] == self.key] # Intercept filters which are calculated values
        hp = self.parent.backward(columns_backward=self.columns_backward, filters_backward=[f for f in filters_backward if f not in self.filters])
        return self.hash(h=hp)

    def fetch(self, verbose):
        tp = self.parent.get(verbose)
        t = tp.append_column(self.key, self.column.get(tp))
        return (filters(t, self.filters) if self.filters else t)

class BooleanMaskNode(BaseNode):
    def __init__(self, parent, mask, cache_obj=None):
        self.parent, self.mask, self.cache_obj = parent, mask, cache_obj
        self.cache = (cache_obj != None)
        self.columns = parent.columns
        self.columns_source, self.columns_forward, self.filters_forward = parent.columns_source, parent.columns_forward, parent.filters_forward
    
    def fetch(self, verbose):
        tp = self.parent.get(verbose)
        t = tp.filter(self.mask.get(tp))
        return t

class FillNanNode(BaseNode):
    def __init__(self, parent, columns, value, cache_obj=None):
        self.parent, self.nan_columns, self.value, self.cache_obj = parent, columns, value, cache_obj
        self.cache = (cache_obj != None)
        self.check(needed=self.nan_columns, reference=parent.columns)
        self.columns = parent.columns

        # Forward propagation of nodes
        self.columns_source, self.columns_forward, self.filters_forward = parent.columns_source, list(set(parent.columns_forward + [c for c in self.nan_columns if c in parent.columns_source])), parent.filters_forward
    
    def fetch(self, verbose):
        t = self.parent.get(verbose)
        for c in self.nan_columns:
            arr = pa.compute.fill_null(t.column(c).combine_chunks(), pa.scalar(self.value))
            t = t.drop([c])
            t = t.append_column(c, arr)
        return t

class CastNode(BaseNode):
    def __init__(self, parent, dtypes, cache_obj=None):
        self.parent, self.dtypes, self.cache_obj = parent, dtypes, cache_obj
        self.cache = (cache_obj != None)
        self.check(needed=self.dtypes.keys(), reference=parent.columns)
        self.columns = parent.columns

        # Forward propagation of nodes
        self.columns_source, self.columns_forward, self.filters_forward = parent.columns_source, list(set(parent.columns_forward + [c for c in list(dtypes.keys()) if c in parent.columns_source])), parent.filters_forward
    
    def fetch(self, verbose):
        t = self.parent.get(verbose)
        for c, tp in self.dtypes.items():
            arr = pa.array(t.column(c).to_numpy().astype(tp))
            t = t.drop([c])
            t = t.append_column(c, arr)
        return t

class DropNode(BaseNode):
    def __init__(self, parent, columns, cache_obj=None):
        self.parent, self.drop_columns, self.cache_obj = parent, (columns if isinstance(columns, list) else [columns]), cache_obj
        self.cache = (cache_obj != None)
        self.check(needed=self.drop_columns, reference=parent.columns)
        self.columns = [c for c in parent.columns if c not in self.drop_columns]

        # Forward propagation of nodes
        self.columns_source, self.columns_forward, self.filters_forward = parent.columns_source, parent.columns_forward, parent.filters_forward
    
    def fetch(self, verbose):
        t = self.parent.get(verbose)
        return t.drop(self.drop_columns)