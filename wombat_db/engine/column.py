import pyarrow as pa
import numpy as np

# Numerical & logical operations 
class ColumnNode():
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
        return self.__class__(key=keyn, required=[r + '.' + key for r in self.required], func=f, depth=self.depth + 1)

    def breed(self, op, other_key, f, required=[], boolean=False):
        key = '(' + self.key + op + other_key + ')'
        return self.__class__(key=key, required=self.required + required, func=f, depth=self.depth + 1, boolean=boolean)

    # Numerical
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
    
    # Comparison operations
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

    # Logical operators
    def __invert__(self):
        f = lambda t: pa.compute.invert(self.get(t))
        return ColumnNode(key='~' + self.key, required=self.required, func=f, depth=self.depth + 1, boolean=True)

    def __and__(self, other):
        f = lambda t: pa.compute.and_(self.get(t), other.get(t))
        return self.breed('&', other.key, f, required=other.required, boolean=True)

    def __or__(self, other):
        f = lambda t: pa.compute.or_(self.get(t), other.get(t))
        return self.breed('|', other.key, f, required=other.required, boolean=True)

    def __xor__(self, other):
        f = lambda t: pa.compute.xor(self.get(t), other.get(t))
        return self.breed('^', other.key, f, required=other.required, boolean=True)

    # Rounding based
    def round(self, decimals=0):
        f = lambda t: pa.array(np.round(self.get(t).to_numpy(), decimals=decimals))
        return ColumnNode(key='round({}, {})'.format(self.key, decimals), required=self.required, func=f, depth=self.depth + 1)

    def ceil(self):
        f = lambda t: pa.array(np.ceil(self.get(t).to_numpy()))
        return ColumnNode(key='ceil({})'.format(self.key), required=self.required, func=f, depth=self.depth + 1)

    def floor(self):
        f = lambda t: pa.array(np.floor(self.get(t).to_numpy()))
        return ColumnNode(key='floor({})'.format(self.key), required=self.required, func=f, depth=self.depth + 1)

    # Casting
    def astype(self, dtype):
        f = lambda t: pa.array(self.get(t).to_numpy().astype(dtype))
        return ColumnNode(key='cast({} as {})'.format(self.key, dtype), required=self.required, func=f, depth=self.depth + 1)
    
    def cast(self, dtype):
        return self.astype(dtype)

    # SQL based
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

    # UDFs
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
