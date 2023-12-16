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
```

`JsonPath.values()` always returns a list of all the values in the given
structure that matched the path. If no items in the structure matched, the
list is empty.

## Syntax

| **Syntax**         | **Description**                                                                                                                                                    |
|--------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `$`                | The root object.                                                                                                                                                   |
| `@`                | The current object.                                                                                                                                                |
| `key`              | Looks up the string in the current dictionary.                                                                                                                     |
| `'key'` or `"key"` | Looks up the string in the current dictionary. Use this for keys with non-alphanumeric characters. You can escape a character inside the quotes using a backslash. |
| `[num]`            | Looks up the `num`th item in the current list. You can also use Python slice syntax, such as `[1:-2]` or `[::2]`                                                   |
| `path1.path2`      | Finds items that match `path2` that are children of items matching `path1`.                                                                                        |
| `path1[path2]`     | Same as `path1.(path2)`                                                                                                                                            |
| `path1..path2`     | Recursively finds items that match `path2` that are descendants of of items matching `path1`.                                                                      |
| `*`                | Returns every item in the current list or every value in the current dict.                                                                                         |
| ``path1 \| path2`` | Finds any items that match `path1` and also any items that match `path2` (union).                                                                                  |
| `path1 & path2`    | Finds any items that match *both* `path1` and `path2` (intersection).                                                                                              |
| `path1 \|\| path2` | If any items match `path1`, this expression returns those items. Otherwise, it returns any items that match `path2` (or).                                          |
| `path1 <- path2`   | Finds any items that match `path1` that have children that match `path2`.                                                                                          |
| `path1.parent()`   | Finds the parents of any items that match `path1`                                                                                                                  |
| `path1.len()`      | Finds the lengths of items that match `path1`.                                                                                                                     |
| `path1.keys()`     | For dicts that `path1`, this returns the keys. For lists, it returns the items in the list. (Currently this just calls `list()` on  the matches).                  |
