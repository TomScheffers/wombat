from wombat.nodes import *
from wombat.sql import parse_sql

# Computation plan (of multiple nodes)
class Plan():
    def __init__(self, node):
        self.database, self.last, self.cache_dict = node.database, node, node.cache_dict

    def __getitem__(self, key):
        if isinstance(key, ColumnNode):
            if key.boolean:
                self.last = BooleanMaskNode(self.last, key, self.cache_dict)
            else:
                raise Exception('Column must be boolean to be used in selection')
        elif key in self.last.columns_available:
            return ColumnNode(key, required=[key])
        else:
            raise Exception("{} not a valid column reference")

    def __setitem__(self, key, value):
        if isinstance(value, ColumnNode):
            self.last = CalculationNode(self.last, key, value, cache_dict=self.cache_dict)
        else:
            raise Exception("Value must be a column node reference")

    def collect(self):
        self.last.backward(columns_backward=self.last.columns_forward, filters_backward=self.last.filters_forward)
        return self.last.get()

    def filter(self, filters):
        self.last = FilterNode(self.last, filters, cache_dict=self.cache_dict)
        return self

    def join(self, right, on):
        if isinstance(right, str):
            plan = self.database.select(right)
            self.last = JoinNode(self.last, plan.last, on, cache_dict=self.cache_dict)
        else:
            self.last = JoinNode(self.last, right.last, on, cache_dict=self.cache_dict)
        return self

    def aggregate(self, by, methods):
        self.last = AggregateNode(self.last, by, methods, cache_dict=self.cache_dict)
        return self

    def rename(self, mapping):
        self.last = SelectionNode(self.last, list(mapping.keys()), aliases=list(mapping.values()), cache_dict=self.cache_dict)
        return self

    def select(self, columns):
        self.last = SelectionNode(self.last, columns, cache_dict=self.cache_dict)
        return self

    def orderby(self, key, ascending=True):
        self.last = OrderNode(self.last, key, ascending, cache_dict=self.cache_dict)
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

# The database class is used to:
# 1. Register tables
# 2. Create a new query plan
# 3. Optimize a query plan
# 4. Execute a query plan
class Engine():
    def __init__(self, cache=True):
        self.cache, self.tables, self.datasets, self.udfs = cache, {}, {}, {}
        self.cache_dict = ({} if cache else None)

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
            return Plan(TableNode(name, self, cache_dict=self.cache_dict))
        elif name in self.datasets.keys():
            return Plan(DatasetNode(name, self, cache_dict=self.cache_dict))
        else:
            raise Exception("{} not in registered tables or datasets".format(name))

    def sql(self, sql):
        # Parse subqueries
        return parse_sql(self, sql)

