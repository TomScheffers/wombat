import re

def analyse_select_part(select):
	aggregate = '(' in select
	pcs = [s for s in select.split(' ') if s != '']
	column = (pcs[0] if not aggregate else pcs[0][pcs[0].find('(') + 1 : pcs[0].find(')')])
	method = (pcs[0][:pcs[0].find('(')] if aggregate else None)
	alias = (pcs[pcs.index('as') + 1] if 'as' in pcs else None)
	return column, alias, method


def parse_subquery(db, sql):
	# 1. FROM -> 2. JOINs -> 3. WHERE -> 4. CALCULATE -> 5. AGGREGATE -> 6. CALCULATE -> 7. HAVING -> 8. SELECT -> 9. ORDER BY -> 10. LIMIT
	sqll = re.sub(r"[\n\t]*", "", sql.lower()) # re.sub(' +', ' ', 
	print(sqll)

	# 0. Split query by keys
	keys = ['select', 'from', 'where', 'group by', 'having', 'order by', 'limit']
	idxs = [sqll.find(k) for k in keys]

	# Gather texts
	keys, idxs = zip(*[(k, i) for k, i in zip(keys, idxs) if i != -1])
	parts = {k:sqll[i + len(k) + 1 : j] for k, i, j in zip(keys, idxs, idxs[1:] + (None,))}

	# 1. Extract JOINs out FROM
	from_sql = parts['from']
	join_keys = ['join', 'inner join']
	join_idxs = [[m.start() for m in re.finditer(k, from_sql.lower())] for k in join_keys]

	# 1a. FROM
	from_end = min([i for fi in join_idxs for i in fi] + [len(from_sql)])
	if re.sub(' ', '', from_sql)[0] == '(':
		plan = parse_subquery(from_sql[1:from_end - 1])
	else:
		from_table = re.sub(' ', '', from_sql[0:from_end])
		plan = db.select(from_table)

	# 2. JOINs
	join_idxs_flat = sorted([i for join_idx in join_idxs for i in join_idx])
	join_parts = {i:from_sql[i:j] for i, j in zip(join_idxs_flat, join_idxs_flat[1:] + [None])}

	for k, join_idx in zip(join_keys, join_idxs):
		for i in join_idx:
			# Analyse the join sql
			join_sql = re.sub(' +', ' ', join_parts[i][len(k) + 1:])
			join_table = [s for s in join_sql.split(' ') if s != ''][0]
			if join_table[0] == '(':
				join_plan = parse_subquery(join_sql)
			else:
				join_plan = db.select(join_table)
			
			# Analyse whether USING / ON is used
			if 'on' in join_sql.lower():
				raise Exception("Not implemented: ON")
			elif 'using' in join_sql.lower():
				on = join_sql[join_sql.find('(') + 1: join_sql.find(')')].replace(' ', '').split(',')

			# Perform JOIN in the plan
			plan = plan.join(join_plan, on=on)
	
	# 3. WHERE
	if 'where' in parts.keys():
		where_sql = parts['where']
		if ' or ' in where_sql:
			raise Exception('Not implemented')
		else:
			where_clauses = where_sql.replace("'", "").split(' and ')
			filt = [tuple(c.split(' ')[:3]) for c in where_clauses]
			print(filt)
			plan = plan.filter(filt)

	# Gather selections
	selections = [analyse_select_part(s) for s in parts['select'].split(',')]
	print(selections)

	# 5. GROUP BY AGGREGATE
	if 'group by' in parts.keys():
		by = parts['group by'].replace(' ', '').split(',')
		methods = {col:method for col, alias, method in selections if method}
		plan = plan.aggregate(by=by, methods=methods)

	# 9. ORDER BY
	if 'order by' in parts.keys():
		orderby = [[v for v in p.split(' ') if v != ''] for p in parts['order by'].split(',')]
		plan = plan.orderby(orderby[0][0], ascending=(orderby[0][1] == 'asc' if len(orderby[0]) > 1 else True))

	return plan

def match_parenthesis(text):
	start, d = [], {}
	for i, c in enumerate(text):
		if c == '(':
			istart.append(i)
		if c == ')':
			try:
				d[istart.pop()] = i
			except IndexError:
				print('Too many closing parentheses')
	return d

def parse_sql(db, sql):
	return parse_subquery(db, sql)