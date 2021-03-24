from wombat_db.engine.nodes import *
from wombat_db.engine.sql import parse_sql
from wombat_db.engine.column import ColumnNode

# Computation plan (of multiple nodes)
class Plan():
    def __init__(self, node):
        self.database, self.last, self.cache_obj = node.database, node, node.cache_obj

    def __getitem__(self, key):
        if isinstance(key, ColumnNode):
            if key.boolean:
                self.last = BooleanMaskNode(self.last, key, self.cache_obj)
            else:
                raise Exception('Column must be boolean to be used in selection')
        elif key in self.last.columns:
            return ColumnNode(key, required=[key])
        else:
            raise Exception("{} not a valid column reference")

    def __setitem__(self, key, value):
        if isinstance(value, ColumnNode):
            self.last = CalculationNode(self.last, key, value, cache_obj=self.cache_obj)
        else:
            raise Exception("Value must be a column node reference")

    def collect(self, verbose=False):
        if verbose:
            print("Columns:", ", ".join(self.last.columns_forward))
        self.last.backward(columns_backward=self.last.columns_forward, filters_backward=self.last.filters_forward)
        return self.last.get(verbose)

    def filter(self, filters):
        self.last = FilterNode(self.last, filters, cache_obj=self.cache_obj)
        return self

    def join(self, right, on):
        if isinstance(right, str):
            plan = self.database.select(right)
            self.last = JoinNode(self.last, plan.last, on, cache_obj=self.cache_obj)
        else:
            self.last = JoinNode(self.last, right.last, on, cache_obj=self.cache_obj)
        return self

    def aggregate(self, by, methods):
        self.last = AggregateNode(self.last, by, methods, cache_obj=self.cache_obj)
        return self

    def rename(self, mapping):
        self.last = SelectionNode(self.last, list(mapping.keys()), aliases=list(mapping.values()), cache_obj=self.cache_obj)
        return self

    def select(self, columns=[]):
        self.last = SelectionNode(self.last, columns, cache_obj=self.cache_obj)
        return self

    def drop(self, columns=[]):
        self.last = DropNode(self.last, columns, cache_obj=self.cache_obj)
        return self

    def orderby(self, key, ascending=True):
        self.last = OrderNode(self.last, key, ascending, cache_obj=self.cache_obj)
        return self

    def fillna(self, columns, value):
        self.last = FillNanNode(self.last, columns, value, cache_obj=self.cache_obj)
        return self

    def cast(self, dtypes):
        self.last = CastNode(self.last, dtypes, cache_obj=self.cache_obj)
        return self

    def udf(self, name, arguments):
        return ColumnNode.udf(name=name, function=self.database.udfs[name], arguments=arguments)

    def plot(self, name):
        from graphviz import Digraph
        dot = Digraph()
        nodes, count = [self.last], 1
        dot.node(str(count), label=self.last.graph_info(), shape='box')
        while nodes:
            if hasattr(nodes[0], 'parent'):
                count += 1
                dot.node(str(count), label=nodes[0].parent.graph_info(), shape='box')
                dot.edge(str(count), str(count - 1))
                nodes[0] = nodes[0].parent
            elif hasattr(nodes[0], 'left'):
                # Right first
                count += 1
                dot.node(str(count), label=nodes[0].right.graph_info(), shape='box')
                dot.edge(str(count), str(count - 1))
                nodes.append(nodes[0].right)

                # Left     
                count += 1   
                dot.node(str(count), label=nodes[0].left.graph_info(), shape='box')
                dot.edge(str(count), str(count - 2))
                nodes[0] = nodes[0].left
            else:
                nodes.pop(0)
        dot.render('plan/{}'.format(name), view=True)
        return

class Cache():
    def __init__(self, max_memory=1e9):
        self.tables, self.importance, self.memory, self.max_memory = {}, {}, 0, max_memory

    def put(self, key, table, weight=1.0):
        self.importance[key] = self.importance.get(key, 0.0) + weight
        if key not in self.tables.keys():
            b = table.nbytes
            while True:
                if self.memory + b < self.max_memory:
                    self.tables[key] = table
                    self.memory += b
                    return
                
                importances = [self.importance[k] for k in self.tables.keys()]
                if not importances:
                    return
                
                if self.importance[key] > min(importances):
                    min_key = importances.index(min(importances))
                    self.memory -= self.tables[min_key]
                    del self.tables[min_key]
                else:
                    return

    def keys(self):
        return self.tables.keys()

    def __getitem__(self, key):
        return self.tables[key]

class Engine():
    def __init__(self, cache_memory=0):
        self.cache, self.tables, self.datasets, self.udfs = (cache_memory > 0), {}, {}, {}
        self.cache_obj = (Cache(max_memory=cache_memory) if self.cache else None)

    def register_table(self, name, table):
        self.tables[name] = table

    def register_dataset(self, name, dataset):
        self.datasets[name] = dataset

    def register_udf(self, name, function):
        self.udfs[name] = function

    def __getitem__(self, key):
        return self.select(key)

    def select(self, name):
        # Return a plan from a source node
        if name in self.tables.keys():
            return Plan(TableNode(name, self, cache_obj=self.cache_obj))
        elif name in self.datasets.keys():
            return Plan(DatasetNode(name, self, cache_obj=self.cache_obj))
        else:
            raise Exception("{} not in registered tables or datasets".format(name))

    def sql(self, sql):
        # Parse subqueries
        return parse_sql(self, sql)
