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

def prepare_connection(conn):
	"""
	#TO INSTANTIATE:
	conn = prepare_connection(sqlite3.connect('example.db', detect_types=sqlite3.PARSE_DECLTYPES))

	#HELPER FUNCS:
	sql = lambda x: conn.execute(x).fetchall()
	to_df = lambda x: pd.read_sql(x, conn)
	"""
	conn.create_function("regexp_matches", 3, regexp_matches)
	conn.create_function('regexp_replace', 3, regexp_replace)
	conn.create_function('split_part', 3, split_part)
	return conn

