import pyarrow as pa
import numpy as np
from pyarrow_ops import join, groupby, filters
import hashlib, json, time

# Computation nodes
class BaseNode():
    def check(self):
        assert all(c in self.columns_available for c in self.columns_forward)

    def columns_bw(self, columns_backward):
        self.columns_backward = [c for c in sorted(list(set(self.columns_forward + columns_backward))) if c in self.columns_available]
        return self.columns_backward

    def hash(self, h=None):
        # We want to add: on, by, columns_backward, table_url
        fields = ['name', 'on', 'filters', 'by', 'methods', 'key', 'ascending', 'columns_backward']
        if not h:
            h = hashlib.sha256()
        obj = {k: v for k,v in self.__dict__.items() if k in fields}
        obj['__name__'] = self.__class__.__name__
        h.update(json.dumps(obj, sort_keys=True).encode())
        self.hash_key = h.hexdigest()
        return h

    def backward(self, columns_backward=[], filters_backward=[]):
        self.columns_bw(columns_backward)
        self.filters = list(set(self.filters_forward + filters_backward))
        hp = self.parent.backward(columns_backward=self.columns_backward, filters_backward=self.filters)
        return self.hash(h=hp)

    def get(self):
        self.time = time.time()
        if self.cache and self.hash_key in self.cache_dict.keys():
            t = self.cache_dict[self.hash_key]
            print("Node: {} Rows: {} Cumulative Time: {:2f} (cached)".format(self.__class__.__name__.ljust(16), str(t.num_rows).ljust(9), time.time() - self.time))
        else:
            t = self.fetch()
            print("Node: {} Rows: {} Cumulative Time: {:2f}".format(self.__class__.__name__.ljust(16), str(t.num_rows).ljust(9), time.time() - self.time))
            if self.cache:
                self.cache_dict[self.hash_key] = t
        return t

# Sources
class TableNode(BaseNode):
    def __init__(self, name, database, cache_dict=None):
        self.name, self.database, self.cache_dict = name, database, cache_dict
        self.cache = (cache_dict != None)
        self.table = self.database.tables[name]

        # Forward propagation of nodes
        self.columns_available, self.columns_forward, self.filters_forward = self.table.column_names, [], []

    def backward(self, columns_backward=[], filters_backward=[]):
        self.columns_bw(columns_backward)
        self.filters = list(set(self.filters_forward + self.filters_backward))
        return self.hash()

    def fetch(self):
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
    def __init__(self, name, database, cache_dict=None):
        self.name, self.database, self.cache_dict = name, database, cache_dict
        self.cache = (cache_dict != None)
        self.dataset = database.datasets[name]

        self.partition_keys = [p.name for p in self.dataset.partitions]
        self.partition_values = [{pk[0]: dp.keys[pk[1]] for pk, dp in zip(p.partition_keys, self.dataset.partitions)} for p in self.dataset.pieces]
        self.meta = self.dataset.pieces[0].get_metadata()

        # Forward propagation of nodes
        self.columns_available, self.columns_forward, self.filters_forward = self.partition_keys + [c['path_in_schema'] for c in self.meta.row_group(0).to_dict()['columns']], [], []

    def backward(self, columns_backward=[], filters_backward=[]):
        self.columns_bw(columns_backward)
        self.filters = list(set(self.filters_forward + filters_backward))
        self.part_filters = list(filter(lambda f: f[0] in self.partition_keys, self.filters))
        self.value_filters = list(filter(lambda f: f[0] not in self.partition_keys, self.filters))
        return self.hash()

    def check(self, partition_value, filters):
        for key, op, value in filters:
            if not part_check(partition_value[key], op, value):
                return False
        return True

    def fetch(self):
        ts = []
        for i, p in enumerate(self.dataset.pieces):
            if self.check(self.partition_values[i], self.part_filters):
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
    def __init__(self, left, right, on, cache_dict=None):
        self.left, self.right, self.on, self.cache_dict = left, right, (on if isinstance(on, list) else [on]), cache_dict
        self.cache = (cache_dict != None)

        # Forward propagation of nodes
        self.columns_available, self.columns_forward = list(set(left.columns_available + right.columns_available)), list(set(left.columns_forward + right.columns_forward + on))
        self.filters_forward = list(set(left.filters_forward + right.filters_forward)) #[f for f in left.filters_forward + right.filters_forward if f[0] in self.on]
        self.check()

    def backward(self, columns_backward=[], filters_backward=[]):
        self.columns_bw(columns_backward)
        self.filters = list(set(self.filters_forward + filters_backward))
        filters_l, filters_r = [f for f in self.filters if f[0] in self.left.columns_available], [f for f in self.filters if f[0] in self.right.columns_available]
        columns_l, columns_r = [c for c in self.columns_backward if c in self.left.columns_available], [c for c in self.columns_backward if c in self.right.columns_available]
        hl = self.left.backward(columns_backward=columns_l, filters_backward=filters_l)
        hr = self.right.backward(columns_backward=columns_r, filters_backward=filters_r)
        hl.update(hr.digest())
        return self.hash(h=hl)

    def fetch(self):
        tl = self.left.get()
        tr = self.right.get()
        on = []
        for i, o in enumerate(self.on):
            l_mmx, r_mmx = column_min_max(tl.column(o).combine_chunks()), column_min_max(tr.column(o).combine_chunks())

            # Skip if there is only 1 value
            if (l_mmx == r_mmx) and (l_mmx[0] == l_mmx[1]):
                # print("Skipping on column:", o)
                continue
            
            # Add column to join condition
            on.append(o)

            # Filter min max values, before joining
            # tlf = filters(tl, [(o, '>=', max(l_mmx[0], r_mmx[0])), (o, '<=', min(l_mmx[1], r_mmx[1]))])
            # trf = filters(tr, [(o, '>=', max(l_mmx[0], r_mmx[0])), (o, '<=', min(l_mmx[1], r_mmx[1]))])
        return join(left=tl, right=tr, on=on)

class FilterNode(BaseNode):
    def __init__(self, parent, filters, cache_dict=None):
        self.parent, self.filters, self.cache_dict = parent, ([filters] if isinstance(filters, tuple) else filters), cache_dict
        self.cache = (cache_dict != None)

        # Forward propagation of nodes
        self.columns_available, self.columns_forward = parent.columns_available, list(set(parent.columns_forward + [f[0] for f in self.filters]))
        self.filters_forward = list(set(parent.filters_forward + self.filters))
        self.check()

    def backward(self, columns_backward=[], filters_backward=[]):
        self.columns_bw(columns_backward)
        self.filters = list(set(self.filters_forward + filters_backward))
        hp = self.parent.backward(columns_backward=self.columns_backward, filters_backward=self.filters)
        # Filter node does not change anything, so can just pass its parents hash
        self.hash_key = hp
        return hp

    def fetch(self):
        return self.parent.get()

class AggregateNode(BaseNode):
    def __init__(self, parent, by, methods, cache_dict=None):
        self.parent, self.by, self.methods, self.cache_dict = parent, (by if isinstance(by, list) else [by]), methods, cache_dict
        self.cache = (cache_dict != None)

        # Forward propagation of nodes
        self.columns_available, self.columns_forward, self.filters_forward = parent.columns_available, list(set(parent.columns_forward + self.by + list(self.methods.keys()))), parent.filters_forward
        self.check()

    def fetch(self):
        tp = self.parent.get()
        return groupby(tp, self.by).agg(self.methods)

class OrderNode(BaseNode):
    def __init__(self, parent, key, ascending, cache_dict=None):
        self.parent, self.key, self.ascending, self.cache_dict = parent, key, ascending, cache_dict
        self.cache = (cache_dict != None)
    
        # Forward propagation of nodes
        self.columns_available, self.columns_forward, self.filters_forward = parent.columns_available, list(set(parent.columns_forward + [self.key])), parent.filters_forward
        self.check()
    
    def fetch(self):
        tp = self.parent.get()
        idxs = pa.compute.sort_indices(tp.column(self.key)).to_numpy()
        return (tp.take(idxs) if self.ascending else tp.take(idxs[::-1]))

class SelectionNode(BaseNode):
    def __init__(self, parent, columns, aliases=[], cache_dict=None):
        self.parent, self.columns, self.aliases, self.cache_dict = parent, columns, aliases, cache_dict
        self.cache = (cache_dict != None)

        # Forward propagation of nodes
        self.mapping = (dict(zip(columns, aliases)) if aliases else {}) 
        self.columns_available, self.columns_forward, self.filters_forward = parent.columns_available + self.aliases, list(set(parent.columns_forward + columns)), parent.filters_forward
        self.check()

    def fetch(self):
        tp = self.parent.get()
        if self.aliases:
            return tp.rename_columns([self.mapping.get(col, col) for col in tp.column_names])
        else:
            return tp.select(self.columns)

# Numerical & logical operations 
class ColumnNode(BaseNode):
    def __init__(self, key, required, func=None, depth=0, boolean=False):
        self.key, self.required, self.depth, self.boolean = key, required, depth, boolean
        self.f = (lambda t: t[self.key] if not func else func)

    def get(self, t):
        f = self.f
        for _ in range(self.depth):
            f = self.f(t)
        return f(t)

    def breed(self, op, other_key, f, required=[], boolean=False):
        key = '(' + self.key + op + other_key + ')'
        return ColumnNode(key=key, required=self.required + required, func=f, depth=self.depth + 1, boolean=boolean)

    def __add__(self, other):
        if isinstance(other, self.__class__):
            f = lambda t: pa.compute.add(self.get(t), other.get(t))
            return self.breed('+', other.key, f, required=other.required)
        else:
            f = lambda t: pa.compute.add(self.get(t), pa.scalar(other))
            return self.breed('+', str(other), f)

    def __sub__(self, other):
        if isinstance(other, self.__class__):
            f = lambda t: pa.compute.subtract(self.get(t), other.get(t))
            return self.breed('-', other.key, f, required=other.required)
        else:
            f = lambda t: pa.compute.subtract(self.get(t), pa.scalar(other))
            return self.breed('-', str(other), f)

    def __mul__(self, other):
        if isinstance(other, self.__class__):
            f = lambda t: pa.compute.multiply(self.get(t), other.get(t))
            return self.breed('*', other.key, f, required=other.required)
        else:
            f = lambda t: pa.compute.multiply(self.get(t), pa.scalar(other))
            return self.breed('*', str(other), f)

    def __truediv__(self, other):
        if isinstance(other, self.__class__):
            f = lambda t: pa.compute.divide(self.get(t), other.get(t))
            return self.breed('/', other.key, f, required=other.required)
        else:
            f = lambda t: pa.compute.divide(self.get(t), pa.scalar(other))
            return self.breed('/', str(other), f)

    def __pow__(self, other):
        if isinstance(other, self.__class__):
            f = lambda t: pa.array(np.power(self.get(t).to_numpy(), other.get(t).to_numpy()))
            return self.breed('^', other.key, f, required=other.required)
        else:
            f = lambda t: pa.array(np.power(self.get(t), other))
            return self.breed('^', str(other), f)
    
    def __lt__(self, other):
        if isinstance(other, self.__class__):
            f = lambda t: pa.compute.less(self.get(t), other.get(t))
            return self.breed('<', other.key, f, required=other.required, boolean=True)
        else:
            f = lambda t: pa.compute.less(self.get(t), pa.scalar(other))
            return self.breed('<', str(other), f, boolean=True)

    def __le__(self, other):
        if isinstance(other, self.__class__):
            f = lambda t: pa.compute.less_equal(self.get(t), other.get(t))
            return self.breed('<', other.key, f, required=other.required, boolean=True)
        else:
            f = lambda t: pa.compute.less_equal(self.get(t), pa.scalar(other))
            return self.breed('<', str(other), f, boolean=True)

    def __gt__(self, other):
        if isinstance(other, self.__class__):
            f = lambda t: pa.compute.greater(self.get(t), other.get(t))
            return self.breed('>', other.key, f, required=other.required, boolean=True)
        else:
            f = lambda t: pa.compute.greater(self.get(t), pa.scalar(other))
            return self.breed('>', str(other), f, boolean=True)

    def __ge__(self, other):
        if isinstance(other, self.__class__):
            f = lambda t: pa.compute.greater_equal(self.get(t), other.get(t))
            return self.breed('>=', other.key, f, required=other.required, boolean=True)
        else:
            f = lambda t: pa.compute.greater_equal(self.get(t), pa.scalar(other))
            return self.breed('>=', str(other), f, boolean=True)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            f = lambda t: pa.compute.equal(self.get(t), other.get(t))
            return self.breed('=', other.key, f, required=other.required, boolean=True)
        else:
            f = lambda t: pa.compute.equal(self.get(t), pa.scalar(other))
            return self.breed('=', str(other), f, boolean=True)

    def __ne__(self, other):
        if isinstance(other, self.__class__):
            f = lambda t: pa.compute.not_equal(self.get(t), other.get(t))
            return self.breed('!=', other.key, f, required=other.required, boolean=True)
        else:
            f = lambda t: pa.compute.not_equal(self.get(t), pa.scalar(other))
            return self.breed('!=', str(other), f, boolean=True)

    def __invert__(self):
        f = lambda t: pa.compute.invert(self.get(t))
        return ColumnNode(key='~' + self.key, required=self.required, func=f, depth=self.depth + 1, boolean=True)

class CalculationNode(BaseNode):
    def __init__(self, parent, key, column, cache_dict=None):
        self.parent, self.key, self.column, self.cache_dict = parent, key, column, cache_dict
        self.cache = (cache_dict != None)
    
        # Forward propagation of nodes
        self.columns_available, self.columns_forward, self.filters_forward = parent.columns_available + [key], parent.columns_forward, parent.filters_forward
        self.check()
    
    def fetch(self):
        tp = self.parent.get()
        t = tp.append_column(self.key, self.column.get(tp))
        return t

class BooleanMaskNode(BaseNode):
    def __init__(self, parent, mask, cache_dict=None):
        self.parent, self.mask, self.cache_dict = parent, mask, cache_dict
        self.cache = (cache_dict != None)
        self.columns_available, self.columns_forward, self.filters_forward = parent.columns_available, parent.columns_forward, parent.filters_forward
    
    def fetch(self):
        tp = self.parent.get()
        t = tp.filter(self.mask.get(tp))
        return t