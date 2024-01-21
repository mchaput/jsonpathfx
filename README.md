# jsonpathfx
Fast, modern, idiosyncratic jsonpath implementation in pure Python

## About

This is a pure Python 3.x implementation of a JSON path language
(there is no real standard syntax for JSON path, so this implements one
among many, but I think the syntax is nice).

It uses a hand-written parser which isn't the cleanest code, but it's
fast compared to some pure-Python jsonpath libraries out there (the
parser is literally thousands of times faster than `jsonpath_ng` for
common inputs).

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
for match in jp.find(data):
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
| `path1 \|\| path2` | If any items match `path1`, this expression returns those items. Otherwise, it returns any items that match `path2` (or).                                     |
| `path1 <- path2`   | Finds any items that match `path1` that have children that match `path2` (contains).                                                                          |
| `path1.parent()`   | Finds the parents of any items that match `path1`                                                                                                             |
| `path1.len()`      | Finds the lengths of items that match `path1`.                                                                                                                |
| `path1.keys()`     | For dicts that match `path1`, this returns the keys.                                                                                                          |
| `path1 > 5`        | Finds matches of `path1` that return true for the given comparison. You can use `==`, `=`, `!=`, `<`, `<=`, `>`, or `>=`.                                     |  
| `type == "car"`    | Compares the matchs to a string. With strings you can use an additional operator `=~` which treats the right-hand string as a regular expression.             |
| `path1 + path2`    | Yields the results of applying an operator (`+`, `-`, `*`, or `/`) between all the matches from `path1` and the first match in `path2`. Only matches numbers. |
| `name:path`        | Binds the _key_ or _index_ that matched in the path to the given name (see "bindings" below)                                                                  |

## Grouping

* Operators have the following relative binding strength, from *loosest* to *tightest* binding:
  * `<-` (contains)
  * Comparisons
  * `&` (intersect)
  * `|` (union)
  * `||` (or)
  * `+` `-`
  * `*` `/`
  * `.` (child)
  * `name:` (bind)
* You can use parentheses (`()`) to group clauses.

## Examples

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
for match in jp.find(my_data):
    print("value=", match.value, "bindings=", match.bindings())
```
