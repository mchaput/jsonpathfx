from __future__ import annotations
import dataclasses
import enum
import re
from typing import Callable, Iterable, NamedTuple, Optional, Sequence, Union

JsonValue = Union[int, float, str, list["JsonValue"], tuple,
dict[str, "JsonValue"], None]


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

    def child(self, child: JsonPath) -> JsonPath:
        if isinstance(self, (This, Root)):
            return child
        elif isinstance(child, This):
            return self
        elif isinstance(child, Root):
            return child
        else:
            return Child(self, child)

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
    def __init__(self, key: str, bracketed=False, pos=0):
        self.key = key
        self.bracketed = bracketed
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


class BracketedKey(Key):
    pass


class Index(JsonPath):
    def __init__(self, ix: Union[int, slice], pos=0):
        self.index = ix
        self.pos = pos
        self.bracketed = True

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


class TokenKind(enum.Enum):
    root = enum.auto()
    this = enum.auto()
    child = enum.auto()
    parent = enum.auto()
    where = enum.auto()
    choice = enum.auto()
    every = enum.auto()
    desc = enum.auto()
    merge = enum.auto()
    intersect = enum.auto()
    key = enum.auto()
    br_key = enum.auto()
    index = enum.auto()
    slice = enum.auto()
    func = enum.auto()
    open_paren = enum.auto()
    close_paren = enum.auto()


class Token(NamedTuple):
    kind: TokenKind
    strings: tuple[str, ...]
    pos: int


# :len
#


token_exprs = {
    TokenKind.func: re.compile(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*[(]"),
    TokenKind.root: re.compile(r"\s*[$]\s*"),
    TokenKind.this: re.compile(r"\s*@\s*"),
    TokenKind.every: re.compile(r"\s*[*]\s*"),
    TokenKind.desc: re.compile(r"\s*[.][.]\s*"),
    TokenKind.child: re.compile(r"\s*[.]\s*"),
    TokenKind.parent: re.compile(r"\s*\^\s*"),
    TokenKind.where: re.compile(r"\s*<-\s*"),
    TokenKind.choice: re.compile(r"\s*[|][|]\s*"),
    TokenKind.merge: re.compile(r"\s*[|]\s*"),
    TokenKind.intersect: re.compile(r"\s*[&]\s*"),
    TokenKind.key: re.compile(r"\s*(\w+)\s*", re.UNICODE),
    TokenKind.index: re.compile(r"\s*\[\s*(-?\d+)\s*]\s*"),
    TokenKind.slice: re.compile(
        r"\s*\[(\s*-?\d+\s*)?:(\s*-?\d+\s*)?(:(-?\d+))?]\s*"),
    TokenKind.br_key: re.compile(r"\s*\[(\w+)\s*]\s*"),
    TokenKind.open_paren: re.compile(r"\s*[(]\s*"),
    TokenKind.close_paren: re.compile(r"\s*[)]\s*"),
}
sq_br_start_expr = re.compile(r"\s*\[\s*(?=['\"])")
sq_br_end_expr = re.compile(r"\s*]")
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
            yield Token(TokenKind.key, (key,), start)
        elif m := sq_br_start_expr.match(text, pos):
            key, pos = lex_string(text, m.end())
            if mm := sq_br_end_expr.match(text, pos):
                yield Token(TokenKind.br_key, (key,), start)
                pos = mm.end()
            else:
                raise ParserError(f"Expected ] at {pos}")
        else:
            for tk, expr in token_exprs.items():
                if m := expr.match(text, pos):
                    yield Token(tk, m.groups(), start)
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
        kind: TokenKind, fn: Callable[[JsonPath, JsonPath], JsonPath]
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


def bracketed_item_reducer(tokens: list[Union[JsonPath, Token]]
                           ) -> Optional[JsonPath]:
    for i in range(len(tokens) - 1, 0, -1):
        token = tokens[i]
        if isinstance(token, (BracketedKey, Index)):
            prev = tokens[i - 1]
            if not isinstance(prev, JsonPath):
                raise ParserError(
                    f"Can't apply key/index to {prev} at {token.pos}")
            left = tokens[:i]
            right = tokens[i:]
            return reduce(left).child(reduce(right))


# The ordering here defines the binding priority of the operators, from loosest
# to tightest
binary_ops: Sequence[Callable[[list[Union[JsonPath, Token]]], JsonPath]] = (
    binary_reducer(TokenKind.choice, Choice),
    binary_reducer(TokenKind.merge, Merge),
    binary_reducer(TokenKind.intersect, Intersect),
    binary_reducer(TokenKind.child, lambda left, right: left.child(right)),
    bracketed_item_reducer,
    binary_reducer(TokenKind.desc, Descendants),
    binary_reducer(TokenKind.where, Where),
)


def reduce(tokens: list[Union[JsonPath, Token]]) -> JsonPath:
    assert tokens
    tk0 = tokens[0]
    if not isinstance(tk0, JsonPath):
        raise ParserError(f"Parser error: {tk0} at {tk0.pos}")
    elif len(tokens) == 1:
        return tk0

    for fn in binary_ops:
        if r := fn(tokens):
            return r

    if not tokens:
        raise ParserError("String reduced to an empty path")
    elif len(tokens) == 1:
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
    i = 0
    open_parens: list[int] = []
    while i < len(tokens):
        token = tokens[i]
        kind = token.kind
        if kind == TokenKind.root:
            tokens[i] = Root(token.pos)
        elif kind == TokenKind.this:
            tokens[i] = This(token.pos)
        elif kind == TokenKind.parent:
            tokens[i] = Parent(token.pos)
        elif kind == TokenKind.every:
            tokens[i] = Every(token.pos)
        elif kind == TokenKind.key:
            tokens[i] = Key(token.strings[0], pos=token.pos)
        elif kind == TokenKind.br_key:
            tokens[i] = BracketedKey(token.strings[0], pos=token.pos)
        elif kind == TokenKind.index:
            ix = slice_index(token.strings[0])
            tokens[i] = Index(ix, pos=token.pos)
        elif kind == TokenKind.slice:
            print("strings=", token.strings)
            start = slice_index(token.strings[0])
            end = slice_index(token.strings[1])
            step = slice_index(token.strings[3])
            tokens[i] = Index(slice(start, end, step), pos=token.pos)
        elif kind == TokenKind.open_paren or kind == TokenKind.func:
            open_parens.append(i)
        elif kind == TokenKind.close_paren:
            if not open_parens:
                raise ParserError(f"Unbalenced close paren at {token.pos}")
            start = open_parens.pop()
            op = tokens[start]
            is_func = op.kind == TokenKind.func
            body = tokens[start + 1:i]
            if not body and not is_func:
                del tokens[start:i + 1]
                i = start
                continue

            repl = [reduce(body)] if body else []
            if is_func:
                maker = built_in_makers[op.strings[0]]
                repl = [maker(*repl)]
            tokens[start:i + 1] = repl
            i = start + 1
            continue
        i += 1

    if not tokens:
        raise ParserError("Parse error: no tokens")
    return reduce(tokens)


