from __future__ import annotations
import enum
import re
from typing import Callable, Iterable, NamedTuple, Optional, Sequence, Union

JsonValue = Union[
    int, float, str, list["JsonValue"], tuple, dict[str, "JsonValue"], None
]


class ParserError(Exception):
    pass


class Datum(NamedTuple):
    value: JsonValue
    parents: tuple[Datum, ...]

    @property
    def root(self) -> Datum:
        if self.parents:
            return self.parents[0]
        else:
            return self

    def push(self, value: JsonValue) -> Datum:
        return Datum(value, self.parents + (self,))


def ensure(value: Union[JsonValue, Datum]) -> Datum:
    if isinstance(value, Datum):
        return value
    else:
        return Datum(value, ())


class JsonPath:
    def __init__(self, pos=0):
        self.pos = pos

    def __repr__(self):
        return f"<{type(self).__name__}>"

    def __eq__(self, other):
        # Good enough for "singletons"; classes with parameters should override
        return type(self) is type(other)

    def __hash__(self):
        # Good enough for "singletons"; classes with parameters should override
        return hash(type(self))

    def find(self, datum: Union[JsonValue, Datum]) -> Iterable[Datum]:
        raise NotImplementedError

    def values(self, data: JsonValue) -> Sequence[JsonValue]:
        return [datum.value for datum in self.find(data)]


class BinaryJsonPath(JsonPath):
    def __init__(self, left: JsonPath, right: JsonPath):
        self.left = left
        self.right = right

    def __repr__(self):
        return f"{type(self).__name__}({self.left!r}, {self.right!r})"

    def __eq__(self, other):
        return (type(self) is type(other) and self.left == other.left and
                self.right == other.right)

    def __hash__(self):
        return hash(type(self)) ^ hash(self.left) ^ hash(self.right)

    @property
    def pos(self):
        return self.left.pos


class Root(JsonPath):
    def find(self, datum: Union[JsonValue, Datum]) -> Iterable[Datum]:
        yield ensure(datum).root


class This(JsonPath):
    def find(self, datum: Union[JsonValue, Datum]) -> Iterable[Datum]:
        yield ensure(datum)


class Parent(JsonPath):
    def find(self, datum: Union[JsonValue, Datum]) -> Iterable[Datum]:
        if ensure(datum).parents:
            yield datum.parents[-1]


class Every(JsonPath):
    def find(self, datum: Union[JsonValue, Datum]) -> Iterable[Datum]:
        datum = ensure(datum)
        v = datum.value
        if isinstance(v, dict):
            v = v.values()
        if hasattr(v, "__iter__"):
            for subval in v:
                yield datum.push(subval)


class Child(BinaryJsonPath):
    @classmethod
    def make(cls, left: JsonPath, right: JsonPath) -> JsonPath:
        if isinstance(left, (This, Root)):
            return right
        elif isinstance(right, This):
            return left
        elif isinstance(right, Root):
            return right
        else:
            return Child(left, right)

    def find(self, datum: Union[JsonValue, Datum]) -> Iterable[Datum]:
        for subdatum in self.left.find(ensure(datum)):
            yield from self.right.find(subdatum)


class Where(BinaryJsonPath):
    def find(self, datum: Union[JsonValue, Datum]) -> Iterable[Datum]:
        datum = ensure(datum)
        for subdatum in self.left.find(datum):
            for _ in self.right.find(subdatum):
                yield subdatum
                break


class Descendants(BinaryJsonPath):
    def find(self, datum: Union[JsonValue, Datum]) -> Iterable[Datum]:
        datum = ensure(datum)

        def match_recursively(d: Datum) -> Iterable[Datum]:
            for match in self.right.find(d):
                yield match

            val = d.value
            if isinstance(val, (list, tuple, dict)):
                subvals: Sequence[JsonValue] = (val.values()
                                                if isinstance(d, dict) else val)
                for subval in subvals:
                    yield from match_recursively(d.push(subval))

        for left_match in self.left.find(datum):
            yield from match_recursively(left_match)


class Choice(BinaryJsonPath):
    def find(self, datum: Union[JsonValue, Datum]) -> Iterable[Datum]:
        datum = ensure(datum)
        left_matched = False
        for match in self.left.find(datum):
            yield match
            left_matched = True
        if not left_matched:
            yield from self.right.find(datum)


# Don't call this Union, that's a type annotation!
class Merge(BinaryJsonPath):
    def find(self, datum: Union[JsonValue, Datum]) -> Iterable[Datum]:
        datum = ensure(datum)
        seen: set[int] = set()
        for match in self.left.find(datum):
            yield match
            seen.add(id(match.value))
        for match in self.right.find(datum):
            if id(match.value) not in seen:
                yield match


class Intersect(BinaryJsonPath):
    def find(self, datum: Union[JsonValue, Datum]) -> Iterable[Datum]:
        datum = ensure(datum)
        left_matches = list(self.left.find(datum))
        seen = set(id(d.value) for d in left_matches)
        for match in self.right.find(datum):
            if id(match.value) in seen:
                yield match


class Key(JsonPath):
    def __init__(self, key: str, pos=0):
        self.key = key
        self.pos = pos

    def __repr__(self):
        return f"{type(self).__name__}({self.key!r})"

    def __eq__(self, other):
        return type(self) is type(other) and self.key == other.key

    def __hash__(self):
        return hash(type(self)) ^ hash(self.key)

    def find(self, datum: Union[JsonValue, Datum]) -> Iterable[Datum]:
        datum = ensure(datum)
        this = datum.value
        if isinstance(this, dict):
            key = self.key
            if key in this:
                yield datum.push(this[key])


class Index(JsonPath):
    def __init__(self, ix: Union[int, slice], pos=0):
        self.index = ix
        self.pos = pos

    def __repr__(self):
        return f"{type(self).__name__}({self.index!r})"

    def __eq__(self, other):
        return type(self) is type(other) and self.index == other.index

    def __hash__(self):
        return hash(type(self)) ^ hash(self.index)

    def find(self, datum: Union[JsonValue, Datum]) -> Iterable[Datum]:
        datum = ensure(datum)
        this = datum.value
        if isinstance(this, (list, tuple)):
            ix = self.index
            try:
                subval = this[ix]
            except IndexError:
                return
            if isinstance(ix, int):
                yield datum.push(subval)
            elif isinstance(ix, slice) and isinstance(subval, (list, tuple)):
                for sv in subval:
                    yield datum.push(sv)


class Func(JsonPath):
    def __init__(self, fn: Callable[[JsonValue], JsonValue],
                 args: Sequence[JsonPath] = (), pos=0):
        self.fn = fn
        self.args = args
        self.pos = pos

    def __repr__(self):
        return f"{type(self).__name__}({self.fn!r})"

    def __eq__(self, other):
        return type(self) is type(other) and self.fn == other.fn

    def __hash__(self):
        return hash(type(self)) ^ hash(self.fn)

    def find(self, datum: Union[JsonValue, Datum]) -> Iterable[Datum]:
        datum = ensure(datum)
        yield datum.push(self.fn(datum.value, *self.args))


def func_maker(fn: Callable[[JsonValue], JsonValue]
               ) -> Callable[[], Func]:
    def maker(*args: JsonPath) -> Func:
        return Func(fn, args)

    return maker


built_in_makers: dict[str, Callable[[], JsonPath]] = {
    "len": func_maker(len),
    "sorted": func_maker(sorted),
    "keys": func_maker(list),
    "parent": Parent,
}


class TKind(enum.Enum):
    root = enum.auto()
    this = enum.auto()
    child = enum.auto()
    i_child = enum.auto()
    parent = enum.auto()
    where = enum.auto()
    choice = enum.auto()
    every = enum.auto()
    desc = enum.auto()
    merge = enum.auto()
    intersect = enum.auto()
    key = enum.auto()
    index = enum.auto()
    slice = enum.auto()
    func = enum.auto()
    open_paren = enum.auto()
    close_paren = enum.auto()
    open_square = enum.auto()
    close_square = enum.auto()


class Token(NamedTuple):
    kind: TKind
    strings: tuple[str, ...]
    source: str
    pos: int


token_exprs = {
    TKind.func: re.compile(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*[(]"),
    TKind.root: re.compile(r"\s*[$]\s*"),
    TKind.this: re.compile(r"\s*@\s*"),
    TKind.every: re.compile(r"\s*[*]\s*"),
    TKind.desc: re.compile(r"\s*[.][.]\s*"),
    TKind.child: re.compile(r"\s*[.]\s*"),
    TKind.parent: re.compile(r"\s*\^\s*"),
    TKind.where: re.compile(r"\s*<-\s*"),
    TKind.choice: re.compile(r"\s*[|][|]\s*"),
    TKind.merge: re.compile(r"\s*[|]\s*"),
    TKind.intersect: re.compile(r"\s*&\s*"),
    TKind.key: re.compile(r"\s*(\w+)\s*", re.UNICODE),
    TKind.index: re.compile(r"\s*\[\s*(-?\d+)\s*]\s*"),
    TKind.slice: re.compile(
        r"\s*\[(\s*-?\d+\s*)?:(\s*-?\d+\s*)?(:(-?\d+))?]\s*"),
    TKind.open_paren: re.compile(r"\s*[(]\s*"),
    TKind.close_paren: re.compile(r"\s*[)]\s*"),
    TKind.open_square: re.compile(r"\s*\[\s*"),
    TKind.close_square: re.compile(r"\s*]\s*"),
}
ws_expr = re.compile(r"\s+")
simple_key_expr = re.compile(r"^\s*(\w+)\s*$", re.UNICODE)
simple_path_expr = re.compile(r"^\s*\w+(\s*[.]\w+\s*)+$", re.UNICODE)


def lex_string(text: str, pos: int) -> tuple[str, int]:
    start_pos = pos
    end_char = text[pos]
    pos += 1
    prev = pos
    output: list[str] = []
    while pos < len(text):
        char = text[pos]
        if char == end_char:
            if prev < pos:
                output.append(text[prev:pos])
            return "".join(output), pos + 1
        elif text[pos] == "\\" and pos < len(text) - 1:
            if prev < pos:
                output.append(text[prev:pos])
            output.append(text[pos + 1])
            pos += 2
            prev = pos
        else:
            pos += 1
    raise ParserError(f"Missing end quote at {start_pos}")


def lex(text: str) -> Iterable[Token]:
    pos = 0
    while pos < len(text):
        start = pos
        if text[pos] in "'\"":
            key, pos = lex_string(text, pos)
            yield Token(TKind.key, (key,), text[start:pos], start)
        else:
            for tk, expr in token_exprs.items():
                if m := expr.match(text, pos):
                    yield Token(tk, m.groups(), m.group(0), start)
                    pos = m.end()
                    break
            else:
                if m := ws_expr.match(text, pos):
                    pos = m.end()
                else:
                    raise ParserError(f"Can't parse {text[pos]!r} at {pos}")


def slice_index(s: Optional[str]) -> Optional[int]:
    if s is not None:
        try:
            return int(s)
        except ValueError:
            raise ParserError(f"Not a valid numeric index: {s!r}")


def binary_reducer(
        kind: TKind, fn: Callable[[JsonPath, JsonPath], JsonPath]
        ) -> Callable[[list[Union[JsonPath, Token]]], Optional[JsonPath]]:
    def reducer(tokens: list[Union[JsonPath, Token]]) -> Optional[JsonPath]:
        for i in range(1, len(tokens)):
            token = tokens[i]
            if isinstance(token, Token) and token.kind == kind:
                left = tokens[:i]
                right = tokens[i + 1:]
                if not right:
                    raise ParserError(
                        f"Expected value after {token} at {token.pos}")
                return fn(reduce(left), reduce(right))
    return reducer


# The ordering in this dict defines the binding priority of the operators, from
# loosest to tightest
BinaryOpType = Callable[[JsonPath, JsonPath], JsonPath]
binary_ops: dict[TKind, BinaryOpType] = {
    TKind.choice: Choice,
    TKind.merge: Merge,
    TKind.intersect: Intersect,
    TKind.child: Child.make,
    TKind.i_child: Child.make,
    TKind.desc: Descendants,
    TKind.where: Where,
}
binary_op_order = tuple(binary_ops)


def reduce(tokens: list[Union[JsonPath, Token]]) -> JsonPath:
    assert tokens
    # Replace atomic tokens with their JsonPath equivalent, and also handle
    # groupings (round and square brackets)
    i = 0
    stack: list[int] = []
    binary_op: Optional[tuple[TKind,]]
    while i < len(tokens):
        token = tokens[i]
        if not isinstance(token, Token):
            i += 1
            continue
        kind = token.kind
        if kind == TKind.root:
            tokens[i] = Root(token.pos)
        elif kind == TKind.this:
            tokens[i] = This(token.pos)
        elif kind == TKind.parent:
            tokens[i] = Parent(token.pos)
        elif kind == TKind.every:
            tokens[i] = Every(token.pos)
        elif kind == TKind.key:
            tokens[i] = Key(token.strings[0], pos=token.pos)
        elif kind == TKind.index:
            ix = slice_index(token.strings[0])
            tokens[i:i + 1] = [Token(TKind.i_child, (), "", token.pos),
                               Index(ix, pos=token.pos)]
        elif kind == TKind.slice:
            start = slice_index(token.strings[0])
            end = slice_index(token.strings[1])
            step = slice_index(token.strings[3])
            tokens[i:i + 1] = [Token(TKind.i_child, (), "", token.pos),
                               Index(slice(start, end, step), pos=token.pos)]
            i = i + 2
            continue
        elif kind in (TKind.open_paren, TKind.func,
                      TKind.open_square):
            stack.append(i)
        elif kind == TKind.close_square:
            open_index = stack.pop()
            open_square = tokens[open_index]
            if open_square.kind != TKind.open_square:
                raise ParserError(f"Unbalanced paren {token} at {token.pos}")
            body = tokens[open_index + 1:i]
            if not body:
                del tokens[open_index:i + 1]
                i = open_index
                continue
            tokens[open_index:i + 1] = [Token(TKind.i_child, (), "", token.pos),
                                        reduce(body)]
            i = open_index + 2
            continue
        elif kind == TKind.close_paren:
            if not stack:
                raise ParserError(f"Unbalenced close paren at {token.pos}")
            open_index = stack.pop()
            open_paren = tokens[open_index]
            if open_paren.kind not in (TKind.func, TKind.open_paren):
                raise ParserError(f"Unbalanced paren {token} at {token.pos}")
            is_func = open_paren.kind == TKind.func
            body = tokens[open_index + 1:i]
            if not body and not is_func:
                del tokens[open_index:i + 1]
                i = open_index
                continue

            repl = [reduce(body)] if body else []
            if is_func:
                maker = built_in_makers[open_paren.strings[0]]
                repl = [maker(*repl)]
            tokens[open_index:i + 1] = repl
            i = open_index + 1
            continue
        i += 1

    if not tokens:
        raise ParserError("String reduced to an empty path")
    tk0 = tokens[0]
    if isinstance(tokens[0], Token) and tk0.kind == TKind.i_child:
        del tokens[0]

    # Find the loosest binary operator in this token list, recursively reduce
    # the sub-lists on either side, then combine them with the op
    bin_op: Optional[tuple[int, int]] = None  # op_index, op_priority
    for i, token in enumerate(tokens):
        if isinstance(token, Token) and token.kind in binary_ops:
            pri = binary_op_order.index(token.kind)
            if bin_op is None or pri < bin_op[1]:
                bin_op = i, pri
    if bin_op:
        op_index = bin_op[0]
        op_token = tokens[op_index]
        operator = binary_ops[op_token.kind]
        left = tokens[:op_index]
        if not left:
            raise ParserError(
                f"Left side of {op_token} is empty at {op_token.pos}")
        right = tokens[op_index + 1:]
        if not right:
            raise ParserError(
                f"Right side of {op_token} is empty at {op_token.pos}")
        return operator(reduce(left), reduce(right))

    if not tokens:
        raise ParserError("String reduced to an empty path")

    if len(tokens) == 1:
        tk0 = tokens[0]
        if isinstance(tk0, JsonPath):
            return tk0
        else:
            raise ParserError(f"Parser error: {tk0} at {tk0.pos}")
    else:
        tk1 = tokens[1]
        raise ParserError(f"Expected operator at {tk1.pos} found {tk1}")


def fast_keypath(p: str) -> JsonPath:
    parts = p.split(".")
    jp = Child(Key(parts[0].strip()), Key(parts[1].strip()))
    for i in range(2, len(parts)):
        jp = Child(jp, Key(parts[i].strip()))
    return jp


def parse(text: str) -> JsonPath:
    if simple_key_expr.match(text):
        return Key(text.strip())
    if simple_path_expr.match(text):
        return fast_keypath(text)

    tokens = list(lex(text))
    if not tokens:
        raise ParserError("Parse error: no tokens")
    return reduce(tokens)
