import functools
import sqlite3
import re
import json
# import jinja2
# import xmltodict
# from datasette import hookimpl
# from Levenshtein import distance as levenshtein_distance

adapt_json = lambda data: (json.dumps(data, sort_keys=True)).encode()
convert_json = lambda blob: json.loads(blob.decode())

sqlite3.register_adapter(dict, adapt_json)
sqlite3.register_adapter(list, adapt_json)
sqlite3.register_adapter(tuple, adapt_json)
sqlite3.register_converter('JSON', convert_json)

def none_on_exception(fn):
	@functools.wraps(fn)
	def inner(*args, **kwargs):
		try:
			return fn(*args, **kwargs)
		except Exception:
			return None

	return inner

@none_on_exception
@functools.lru_cache(maxsize=128)
def regexp_matches(txt, pattern, return_type):
	matches = re.findall(pattern, txt)
	if matches:
		return json.dumps(matches) if return_type =='list' else json.dumps(dict(matches=matches))

@none_on_exception
@functools.lru_cache(maxsize=128)
def regexp_replace(txt, pattern, replacement):
	return re.sub(pattern, replacement, txt)

@none_on_exception
@functools.lru_cache(maxsize=128)
def split_part(txt, char, part_index):
	return txt.split(char)[part_index-1]

# @none_on_exception
# @functools.lru_cache(maxsize=128)
# def parse_sql(sql_text):
# 	try:
# 		return json.dumps([str(stmt) for stmt in sqlparse.split(sql_text)])
# 	except Exception as e:
# 		print(e)

@none_on_exception
@functools.lru_cache(maxsize=128)
def remove_blank_lines(p_txt):
	return '\n'.join([t for t in p_txt.split('\n') if t != ''])

# @none_on_exception
# @functools.lru_cache(maxsize=128)
# def levenshtein(str1, str2):
# 	return levenshtein_distance(str1, str2)

@none_on_exception
@functools.lru_cache(maxsize=128)
def crush_txt(txt):
	return txt.replace('\n','').replace('\t','').replace(' ','') 

# @hookimpl(trylast=True)
# def render_cell(value):
#     if not isinstance(value, str):
#         return None
#     stripped = value.strip()
#     if not (
#         (stripped.startswith("{") and stripped.endswith("}"))
#         or (stripped.startswith("[") and stripped.endswith("]"))
#     ):
#         return None
#     try:
#         data = json.loads(value)
#     except ValueError:
#         return None
#     return jinja2.Markup(
#         '<pre>{data}</pre>'.format(
#             data=jinja2.escape(json.dumps(data, indent=4))
#         )
#     )

# @hookimpl
def prepare_connection(conn):
	"""
	#TO INSTANTIATE:
	conn = prepare_connection(sqlite3.connect('example.db', detect_types=sqlite3.PARSE_DECLTYPES))

	#HELPER FUNCS:
	sql = lambda x: conn.execute(x).fetchall()
	to_df = lambda x: pd.read_sql(x, conn)
	"""
	conn.create_function("regexp_matches", 3, regexp_matches)
	conn.create_function('remove_blank_lines', 1, remove_blank_lines)
	conn.create_function('regexp_replace', 3, regexp_replace)
	conn.create_function('split_part', 3, split_part)
	# conn.create_function('levenshtein', 2, levenshtein)
	conn.create_function('crush_txt', 1, crush_txt)	
	return conn

