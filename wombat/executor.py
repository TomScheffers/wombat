from wombat.nodes import *

# Computation plan (of multiple nodes)
class Plan():
    def __init__(self, node):
        self.database, self.last, self.cache_dict = node.database, node, node.cache_dict

    def __getitem__(self, key):
        if key in self.last.columns_available:
            return ColumnNode(key, required=[key])
        else:
            raise Exception("{} not a valid column reference")

    def __setitem__(self, key, value):
        if isinstance(value, ColumnNode):
            self.last = CalculationNode(self.last, key, value, self.cache_dict)
        else:
            raise Exception("Value must be a column node reference")

    def collect(self):
        self.last.backward(columns_backward=self.last.columns_forward, filters_backward=self.last.filters_forward)
        return self.last.get()

    def filter(self, filters):
        self.last = FilterNode(self.last, filters, self.cache_dict)
        return self

    def join(self, right, on):
        if isinstance(right, str):
            plan = self.database.select(right)
            self.last = JoinNode(self.last, plan.last, on, self.cache_dict)
        else:
            self.last = JoinNode(self.last, right.last, on, self.cache_dict)
        return self

    def groupby(self, by):
        self.last = GroupbyNode(self.last, by, self.cache_dict)
        return self

    def agg(self, methods):
        self.last = AggregateNode(self.last, methods, self.cache_dict)
        return self

    def orderby(self, key, ascending=True):
        self.last = OrderNode(self.last, key, ascending, self.cache_dict)
        return self

# The database class is used to:
# 1. Register tables
# 2. Create a new query plan
# 3. Optimize a query plan
# 4. Execute a query plan
class Database():
    def __init__(self, cache=True):
        self.cache, self.tables, self.datasets = cache, {}, {}
        self.cache_dict = ({} if cache else None)

    def register_table(self, name, table):
        self.tables[name] = table

    def register_dataset(self, name, dataset):
        self.datasets[name] = dataset

    def __getitem__(self, key):
        return self.select(key)

    def select(self, name):
        # Return a plan from a source node
        if name in self.tables.keys():
            return Plan(TableNode(name, self, cache_dict=self.cache_dict))
        elif name in self.datasets.keys():
            return Plan(DatasetNode(name, self, cache_dict=self.cache_dict))
        else:
            raise Exception("{} not in registered tables or datasets".format(name))

