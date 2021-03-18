import pyarrow as pa
import numpy as np
from wombat.ops import join, groupby, filters
import hashlib, json, time

# Computation nodes
class BaseNode():
    def check(self):
        assert all(c in self.columns_available for c in self.columns_forward)

    def columns_bw(self, columns_backward):
        self.columns_backward = [c for c in sorted(list(set(self.columns_forward + columns_backward))) if c in self.columns_available]
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
        filters = list(set(self.filters_forward + filters_backward))
        hp = self.parent.backward(columns_backward=self.columns_backward, filters_backward=filters)
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

        # Forward propagation of nodes
        columns = self.table.column_names
        self.columns_available, self.columns_forward, self.filters_forward = columns + list(set([c.split('.')[0] for c in columns if '.' in c])), [], []

    def backward(self, columns_backward=[], filters_backward=[]):
        self.columns_bw(columns_backward)
        self.filters = list(set(self.filters_forward + self.filters_backward))
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

        # Forward propagation of nodes
        columns = self.partition_keys + [c['path_in_schema'] for c in self.meta.row_group(0).to_dict()['columns']]
        self.columns_available, self.columns_forward, self.filters_forward = columns + list(set([c.split('.')[0] for c in columns if '.' in c])), [], []

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

    def fetch(self, verbose):
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
    def __init__(self, left, right, on, cache_obj=None):
        self.left, self.right, self.on, self.cache_obj = left, right, (on if isinstance(on, list) else [on]), cache_obj
        self.cache = (cache_obj != None)

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

        # Forward propagation of nodes
        self.columns_available, self.columns_forward = parent.columns_available, list(set(parent.columns_forward + [f[0] for f in self.filters]))
        self.filters_forward = list(set(parent.filters_forward + self.filters))
        self.check()

    def backward(self, columns_backward=[], filters_backward=[]):
        self.columns_bw(columns_backward)
        filters = list(set(self.filters_forward + filters_backward))
        hp = self.parent.backward(columns_backward=self.columns_backward, filters_backward=filters)
        self.hash_key = hp.hexdigest() # Filter node does not change anything, so can just pass its parents hash
        return hp

    def fetch(self, verbose):
        return self.parent.get(verbose)

class AggregateNode(BaseNode):
    def __init__(self, parent, by, methods, cache_obj=None):
        self.parent, self.by, self.methods, self.filters, self.cache_obj = parent, (by if isinstance(by, list) else [by]), methods, [], cache_obj
        self.cache = (cache_obj != None)

        # Forward propagation of nodes
        self.columns_available, self.columns_forward, self.filters_forward = parent.columns_available, list(set(parent.columns_forward + self.by + list(self.methods.keys()))), parent.filters_forward
        self.check()
    
    def backward(self, columns_backward=[], filters_backward=[]):
        self.columns_bw(columns_backward)
        filters = list(set(self.filters_forward + filters_backward))
        self.filters = [f for f in filters if f[0] in self.methods.keys()] # Intercept filters which are aggregate values
        hp = self.parent.backward(columns_backward=self.columns_backward, filters_backward=[f for f in filters if f not in self.filters])
        return self.hash(h=hp)

    def fetch(self, verbose):
        tp = self.parent.get(verbose)
        t = groupby(tp, self.by).agg(self.methods)
        return (filters(t, self.filters) if self.filters else t)

class OrderNode(BaseNode):
    def __init__(self, parent, key, ascending, cache_obj=None):
        self.parent, self.key, self.ascending, self.cache_obj = parent, key, ascending, cache_obj
        self.cache = (cache_obj != None)
    
        # Forward propagation of nodes
        self.columns_available, self.columns_forward, self.filters_forward = parent.columns_available, list(set(parent.columns_forward + [self.key])), parent.filters_forward
        self.check()
    
    def fetch(self, verbose):
        tp = self.parent.get(verbose)
        idxs = pa.compute.sort_indices(tp.column(self.key)).to_numpy()
        return (tp.take(idxs) if self.ascending else tp.take(idxs[::-1]))

class SelectionNode(BaseNode):
    def __init__(self, parent, columns, aliases=[], cache_obj=None):
        self.parent, self.columns, self.aliases, self.cache_obj = parent, columns, aliases, cache_obj
        self.cache = (cache_obj != None)

        # Forward propagation of nodes
        self.mapping = (dict(zip(columns, aliases)) if aliases else {}) 
        self.columns_available, self.columns_forward, self.filters_forward = parent.columns_available + columns + aliases, list(set(parent.columns_forward + columns)), parent.filters_forward
        self.check()

    def fetch(self, verbose):
        tp = self.parent.get(verbose)
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

    def __getitem__(self, key):
        keyn = self.key + '[' + str(key) + ']'
        if isinstance(key, str):
            f = lambda t: self.get(t).combine_chunks().field(key) # Reference to a Struct Column
        else:
            raise Exception("__getitem__ currently only defined for struct fields")
        return ColumnNode(key=keyn, required=[r + '.' + key for r in self.required], func=f, depth=self.depth + 1)

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
            return self.breed('**', other.key, f, required=other.required)
        else:
            f = lambda t: pa.array(np.power(self.get(t), other))
            return self.breed('**', str(other), f)
    
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

    def greatest(self, other):
        if isinstance(other, self.__class__):
            f = lambda t: pa.array(np.maximum(self.get(t).to_numpy(), other.get(t).to_numpy()))
            return ColumnNode(key='greatest({}, {})'.format(self.key, other.key), required=self.required + other.required, func=f, depth=self.depth + 1)
        else:
            f = lambda t: pa.array(np.maximum(self.get(t).to_numpy(), other))
            return ColumnNode(key='greatest({}, {})'.format(self.key, other), required=self.required, func=f, depth=self.depth + 1)

    def least(self, other):
        if isinstance(other, self.__class__):
            f = lambda t: pa.array(np.minimum(self.get(t).to_numpy(), other.get(t).to_numpy()))
            return ColumnNode(key='least({}, {})'.format(self.key, other.key), required=self.required + other.required, func=f, depth=self.depth + 1)
        else:
            f = lambda t: pa.array(np.minimum(self.get(t).to_numpy(), other))
            return ColumnNode(key='least({}, {})'.format(self.key, other), required=self.required, func=f, depth=self.depth + 1)

    def clip(self, a_min=None, a_max=None):
        f = lambda t: pa.array(np.clip(self.get(t).to_numpy(), a_min, a_max))
        return ColumnNode(key=self.key + '.clip({}, {})'.format(a_min, a_max), required=self.required, func=f, depth=self.depth + 1)

    def fillna(self, value):
        f = lambda t: pa.compute.fill_null(self.get(t), pa.scalar(other))
        return ColumnNode(key='{}.fillna({})'.format(self.key, other), required=self.required, func=f, depth=self.depth + 1)

    def coalesce(self, other):
        if isinstance(other, self.__class__):
            f = lambda t: pa.array(np.where(np.isnan(self.get(t)), other.get(t), self.get(t)))
            return ColumnNode(key='coalesce({}, {})'.format(self.key, other.key), required=self.required + other.required, func=f, depth=self.depth + 1)
        else:
            f = lambda t: pa.compute.fill_null(self.get(t), pa.scalar(other))
            return ColumnNode(key='coalesce({}, {})'.format(self.key, other), required=self.required, func=f, depth=self.depth + 1)

    @classmethod
    def udf(cls, name, function, arguments):
        if isinstance(arguments, dict):
            f = lambda t: function(**{k: (v.get(t) if isinstance(v, ColumnNode) else v) for k,v in arguments.items()})
            keys = ', '.join([(v.key if isinstance(v, ColumnNode) else str(v)) for v in arguments.values()])
            depth = max([0] + [v.depth for v in arguments.values() if isinstance(v, ColumnNode)]) + 1
            required = list(set([r for v in arguments.values() if isinstance(v, ColumnNode) for r in v.required]))
        else:
            if isinstance(arguments, ColumnNode):
                f = lambda t: function(arguments.get(t))
                keys, depth, required = arguments.key, arguments.depth + 1, arguments.required
            else:
                f = lambda t: function(arguments)
                keys, depth, required = str(arguments), 0, []
        return cls(key=name + '(' + keys + ')', required=required, func=f, depth=depth)

class CalculationNode(BaseNode):
    def __init__(self, parent, key, column, cache_obj=None):
        self.parent, self.key, self.calculation, self.column, self.cache_obj = parent, key, column.key, column, cache_obj
        self.cache = (cache_obj != None)
    
        # Forward propagation of nodes
        self.columns_available, self.columns_forward, self.filters_forward = parent.columns_available + [key], parent.columns_forward + column.required, parent.filters_forward
        self.check()
    
    def fetch(self, verbose):
        tp = self.parent.get(verbose)
        t = tp.append_column(self.key, self.column.get(tp))
        return t

class BooleanMaskNode(BaseNode):
    def __init__(self, parent, mask, cache_obj=None):
        self.parent, self.mask, self.cache_obj = parent, mask, cache_obj
        self.cache = (cache_obj != None)
        self.columns_available, self.columns_forward, self.filters_forward = parent.columns_available, parent.columns_forward, parent.filters_forward
    
    def fetch(self, verbose):
        tp = self.parent.get(verbose)
        t = tp.filter(self.mask.get(tp))
        return t