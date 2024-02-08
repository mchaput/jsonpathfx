# jsonpathfx

Fast, modern, idiosyncratic jsonpath implementation in pure Python

## About

This is a pure Python 3.x implementation of a JSON path language
(there is no real standard syntax for JSON path, so this implements one
among many, but I think the syntax is nice).

It uses a proper, extensivle parser, and it's fast compared to some pure-Python
jsonpath libraries out there (the parser is literally thousands of times faster
than `jsonpath_ng` for common inputs).

## API

```python
from jsonpathfx import parse

# Parse a JSON path string into a `jsonpathfx.JsonPath` object
jp = parse("foo.bar")

# Use the JsonPath object to search a JSON-like structure
assert jp.values({"foo": {"bar": 10, "baz": 20}}) == [10]

# You can get the matches as a generator instead of a list
for value in jp.itervalues({"foo": {"bar": 10, "baz": 20}}):
    print(value)

# You can get Match objects with a few useful methods if needed
for match in jp._find(data):
    print("The value of this match is:", match.value)
    print("The path to this match is:", match.path())
    print("The bound key values for this match are:", match.bindings())
```

`JsonPath.values()` always returns a list of all the values in the given
structure that matched the path. If no items in the structure matched, the
list is empty.

## Syntax

| **Syntax**         | **Description**                                                                                                                                               |
|--------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `$`                | The root object.                                                                                                                                              |
| `@`                | The current object.                                                                                                                                           |
| `key`              | Looks up the string in the current dictionary.                                                                                                                |
| `'key'` or `"key"` | Looks up the string in the current dictionary. Use this for keys with non-alphanumeric characters. You can escape characters using backslash.                 |
| `[num]`            | Looks up the `num`th item in the current list. You can also use Python slice syntax, such as `[1:-2]` or `[::2]`                                              |
| `path1.path2`      | Finds items that match `path2` that are children of items matching `path1`.                                                                                   |
| `path1[path2]`     | Same as `path1.(path2)`                                                                                                                                       |
| `path1..path2`     | Recursively finds items that match `path2` that are descendants of of items matching `path1`.                                                                 |
| `*`                | Returns every item in the current list or every value in the current dict.                                                                                    |
| ``path1 \| path2`` | Finds any items that match `path1` and also any items that match `path2` (union).                                                                             |
| `path1 & path2`    | Finds any items that match *both* `path1` and `path2` (intersection).                                                                                         |
| `path1 \|\| path2` | If any items match `path1`, this expression returns those items. Otherwise, it returns any items that match `path2` (logical or).                             |
| `path1 && path2`   | If _both_ expressions match _at least one_ item, yields matches from `path2` (logical and)                                                                    |
| `path1 ! path2`    | Matches results from `path1` if they don't match `path2`                                                                                                      |
| `{path}`           | Matches if the current item has children that match `path` (contains).                                                                                        |
| `parent()`         | Matches the parent of the current item.                                                                                                                       |
| `len()`            | If the current item is an array or object, returns its length.                                                                                                |
| `keys()`           | If the current item is an object, yields its keys.                                                                                                            |
| `items()`          | If the current item is an object, yields `["key", value]` pairs for each item in the object.                                                                  |
| `lookup(path)`     | If the current item is an object, and `path` matches a string, looks that string key up in the object. Also works with a list if `path` matches a number.     |
| `int()`            | Converts the current item to an integer. Only matches if the current item can be converted to an int.                                                         |
| `float()`          | Converts the current item to a float. Only matches if the current item can be converted to a float.                                                           |
| `path1 > 5`        | Finds matches of `path1` that return true for the given comparison. You can use `==`, `=`, `!=`, `<`, `<=`, `>`, or `>=`.                                     |  
| `type == "car"`    | Compares the matchs to a string. With strings you can use an additional operator `=~` which treats the right-hand string as a regular expression.             |
| `path1 + path2`    | Yields the results of applying an operator (`+`, `-`, `*`, or `/`) between all the matches from `path1` and the first match in `path2`. Only matches numbers. |
| `name:path`        | Binds the _key_ or _index_ that matched in the path to the given name (see "bindings" below)                                                                  |
| `%varname`         | Yields the value of the named variable. The variable can be a previous binding or a value from the dict passed to the `env` keyword argument.                 |

You can use Python-style line comments, which may be useful for "verbose"
multi-line path definitions:

```python
p = parse("""
foo.    # Lookup the 'foo' key at the root level
k:*.    # Try every key, bind the matching one to 'k'
bar     # Lookup the bar key
""")
```

## Grouping

* Operators have the following relative binding strength, from *loosest* to *tightest* binding:
  * `()` `{}` (group, contains)
  * `&` (intersect)
  * `|` (union)
  * `||` `&&` (logic operators)
  * `==` `!=` `<` `<=` `>` `>=` (comparisons)
  * `+` `-`
  * `*` `/`
  * `.` (child)
  * `name:` (bind)
* You can use parentheses (`()`) to group clauses.

## Examples


## Comparisons

The left and right hand side of a comparison are treated specially in some ways.

* A comparison yields items from the _left_ side (for which the comparison
  returns true).
* A quoted string on thr _right_ side is treated as a string to compare to,
  instead of a key match as it would be normally. 

## Filtering with {} and comparisons

It's very useful to combine `{}` (contains) syntax with comparisons to filter
items in an array or object. This is often in a form like `*{x > 5}` (find all
objects with a key `x` of value greater than 5).

For example:

```python
from jsonpathfx import parse

doc = {
    "things": [
        {"type": "car", "color": "red", "size": 5, "id": "a"},
        {"type": "boat", "color": "blue", "size": 2, "id": "b"},
        {"type": "car", "color": "blue", "size": 3, "id": "c"},
        {"type": "boat", "color": "red", "size": 6, "id": "d"},
    ]
}
# Find IDs of things where color == "red"
p = parse("things.*{color == 'red'}.id")
assert p.values(doc) == ["a", "d"]
# Find IDs of boats that are red
p = parse("things.*{type == 'boat' && color == 'red'}.id")
assert p.values(doc) == ["d"]
```

## Bindings

Sometimes it's useful to know which key/index of several options inside a path
actually matched  for each result. For example, if you have a path such as:

```
geometry.(points|vertices|faces).rows.*
```

...You might want to know if it was under a `points`, `vertices`, or `faces`
key. To get this information, you can _bind_ that key expression to a name such
as `component`:

```
geometry.component:(points|vertices|faces).rows.*
```

Then, you can retrieve the bindings for each match from the `Match` objects
returned by `JsonPath.find()`:

```python
from jsonpathfx import parse

jp = parse("geometry.component:(points|vertices|faces).rows.*")
for match in jp._find(my_data):
    print("value=", match.value, "bindings=", match.bindings())
```


## Using variables


