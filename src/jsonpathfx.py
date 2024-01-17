from __future__ import annotations
import enum
import operator
import re
from typing import Callable, Iterable, NamedTuple, Optional, Sequence, Union

JsonValue = Union[
    int, float, str, list["JsonValue"], tuple, dict[str, "JsonValue"], None
]


class ParserError(Exception):
    pass


class Match(NamedTuple):
    value: JsonValue
    key: str | int | None
    parents: tuple[Match, ...]
    name: Optional[str] = None

    @property
    def root(self) -> Match:
        if self.parents:
            return self.parents[0]
        else:
            return self

    def push(self, match: str|int|None, value: JsonValue, name: str = None
             ) -> Match:
        return Match(value, match, self.parents + (self,), name)

    def path(self) -> Sequence[str|int|None]:
        return tuple(p.key for p in self.parents[1:]) + (self.key,)

    def bindings(self) -> dict[str, JsonValue]:
        if self.parents:
            bindings = self.parents[-1].bindings()
        else:
            bindings = {}
        if self.name:
            bindings[self.name] = self.key
        return bindings


def ensure(value: Union[JsonValue, Match]) -> Match:
    if isinstance(value, Match):
        return value
    else:
        return Match(value, None, ())


class JsonPath:
    def __init__(self, *, start=0, end=0):
        self._start = start
        self._end = end

    def __repr__(self):
        return f"<{type(self).__name__}>"

    def __eq__(self, other):
        # Good enough for "singletons"; classes with parameters should override
        return type(self) is type(other)

    def __hash__(self):
        # Good enough for "singletons"; classes with parameters should override
        return hash(type(self))

    def find(self, match: Union[JsonValue, Match]) -> Iterable[Match]:
        raise NotImplementedError

    def values(self, data: JsonValue) -> Sequence[JsonValue]:
        return list(self.itervalues(data))

    def itervalues(self, data: JsonValue) -> Iterable[JsonValue]:
        return (match.value for match in self.find(data))

    @property
    def start(self) -> int:
        return self._start

    @property
    def end(self) -> int:
        return self._end


class BinaryJsonPath(JsonPath):
    def __init__(self, left: JsonPath, right: JsonPath):
        super().__init__()
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
    def start(self) -> int:
        return self.left.start

    @property
    def end(self) -> int:
        return self.right.end


class Root(JsonPath):
    def find(self, match: Union[JsonValue, Match]) -> Iterable[Match]:
        yield ensure(match).root


class This(JsonPath):
    def find(self, match: Union[JsonValue, Match]) -> Iterable[Match]:
        match = ensure(match)
        yield match.push(match.value, match)


class Parent(JsonPath):
    def find(self, match: Union[JsonValue, Match]) -> Iterable[Match]:
        if ensure(match).parents:
            yield match.parents[-1]


class Every(JsonPath):
    def find(self, match: Union[JsonValue, Match]) -> Iterable[Match]:
        match = ensure(match)
        v = match.value
        if isinstance(v, dict):
            for key, value in v.items():
                yield match.push(key, value)
        else:
            try:
                iterator = iter(v)
            except TypeError:
                return
            else:
                for i, subval in enumerate(iterator):
                    yield match.push(i, subval)


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

    def find(self, match: Union[JsonValue, Match]) -> Iterable[Match]:
        for subm in self.left.find(ensure(match)):
            yield from self.right.find(subm)


class Where(BinaryJsonPath):
    def find(self, match: Union[JsonValue, Match]) -> Iterable[Match]:
        match = ensure(match)
        for subm in self.left.find(match):
            for _ in self.right.find(subm):
                yield subm
                break


class Descendants(BinaryJsonPath):
    def find(self, match: Union[JsonValue, Match]) -> Iterable[Match]:
        match = ensure(match)

        def match_recursively(this: Match) -> Iterable[Match]:
            for m in self.right.find(this):
                yield m

            val = this.value
            if isinstance(val, dict):
                for key, subval in val.items():
                    yield from match_recursively(this.push(key, subval))
            elif isinstance(val, (list, tuple)):
                for i, subval in enumerate(val):
                    yield from match_recursively(this.push(i, subval))

        for left_match in self.left.find(match):
            yield from match_recursively(left_match)


class Choice(BinaryJsonPath):
    def find(self, match: Union[JsonValue, Match]) -> Iterable[Match]:
        match = ensure(match)
        left_matched = False
        for match in self.left.find(match):
            yield match
            left_matched = True
        if not left_matched:
            yield from self.right.find(match)


# Don't call this Union, that's a type annotation!
class Merge(BinaryJsonPath):
    def find(self, match: Union[JsonValue, Match]) -> Iterable[Match]:
        match = ensure(match)
        seen: set[int] = set()
        for subm in self.left.find(match):
            yield subm
            seen.add(id(subm.value))
        for subm in self.right.find(match):
            if id(subm.value) not in seen:
                yield subm


class Intersect(BinaryJsonPath):
    def find(self, match: Union[JsonValue, Match]) -> Iterable[Match]:
        match = ensure(match)
        left_matches = list(self.left.find(match))
        seen = set(id(d.value) for d in left_matches)
        for match in self.right.find(match):
            if id(match.value) in seen:
                yield match


class Key(JsonPath):
    def __init__(self, key: str, *, start=0, end=0):
        super().__init__(start=start, end=end)
        self.key = key

    def __repr__(self):
        return f"{type(self).__name__}({self.key!r})"

    def __eq__(self, other):
        return type(self) is type(other) and self.key == other.key

    def __hash__(self):
        return hash(type(self)) ^ hash(self.key)

    def find(self, match: Union[JsonValue, Match]) -> Iterable[Match]:
        match = ensure(match)
        this = match.value
        if isinstance(this, dict):
            try:
                value = this[self.key]
            except KeyError:
                return
            else:
                yield match.push(self.key, value)


class Index(JsonPath):
    def __init__(self, ix: Union[int, slice], start=0, end=0):
        super().__init__(start=start, end=end)
        self.index = ix

    def __repr__(self):
        return f"{type(self).__name__}({self.index!r})"

    def __eq__(self, other):
        return type(self) is type(other) and self.index == other.index

    def __hash__(self):
        return hash(type(self)) ^ hash(self.index)

    def find(self, match: Union[JsonValue, Match]) -> Iterable[Match]:
        match = ensure(match)
        this = match.value
        if isinstance(this, (list, tuple)):
            ix = self.index
            if isinstance(ix, int):
                try:
                    yield match.push(ix, this[ix])
                except IndexError:
                    return
            elif isinstance(ix, slice):
                start, stop, step = ix.indices(len(this))
                for i in range(start, stop, step):
                    yield match.push(i, this[i])


class Bind(JsonPath):
    def __init__(self, name: str, child: JsonPath, start=0, end=0):
        super().__init__(start=start, end=end)
        self.name = name
        self.child = child

    def __repr__(self):
        return f"{type(self).__name__}({self.name})"

    def __eq__(self, other):
        return type(self) is type(other) and self.name == other.name

    def __hash__(self):
        return hash(type(self)) ^ hash(self.name)

    def find(self, match: Union[JsonValue, Match]) -> Iterable[Match]:
        match = ensure(match)
        for found in self.child.find(match):
            yield match.push(found.key, found.value, self.name)


class Func(JsonPath):
    def __init__(self, fn: Callable[[JsonValue], JsonValue],
                 args: Sequence[JsonPath] = (), start=0, end=0):
        super().__init__(start=start, end=end)
        self.fn = fn
        self.args = args

    def __repr__(self):
        return f"{type(self).__name__}({self.fn!r})"

    def __eq__(self, other):
        return type(self) is type(other) and self.fn == other.fn

    def __hash__(self):
        return hash(type(self)) ^ hash(self.fn)

    def find(self, match: Union[JsonValue, Match]) -> Iterable[Match]:
        match = ensure(match)
        yield match.push(None, self.fn(match.value, *self.args))


class Comparison(JsonPath):
    comparisons: dict[str, Callable[[JsonValue, JsonValue], JsonValue]] = {
        '!=': operator.ne,
        '==': operator.eq,
        '=': operator.eq,
        '<=': operator.le,
        '<': operator.lt,
        '>=': operator.ge,
        '>': operator.gt,
        '=~': lambda s, expr: bool(isinstance(s, str) and re.search(expr, s)),
    }

    def __init__(self, op_name: str, value: JsonValue, start=0, end=0):
        super().__init__(start=start, end=end)
        self.op_name = op_name
        self.op = self.comparisons[op_name]
        self.val = value

    def __repr__(self):
        return f"{type(self).__name__}({self.op_name!r}, {self.val!r})"

    def __hash__(self):
        return hash(type(self)) ^ hash(self.op) ^ hash(self.val)

    def __eq__(self, other):
        return (type(self) is type(other) and self.op == other.op and
                self.val == other.val)

    def find(self, match: Union[JsonValue, Match]) -> Iterable[Match]:
        match = ensure(match)
        try:
            if self.op(match.value, self.val):
                yield match
        except TypeError:
            pass


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
    implicit_child = enum.auto()
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
    compare = enum.auto()
    open_paren = enum.auto()
    close_paren = enum.auto()
    open_square = enum.auto()
    close_square = enum.auto()
    bind = enum.auto()


class Token(NamedTuple):
    kind: TKind
    strings: tuple[str, ...]
    source: str
    start: int
    end: int


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
number_expr = re.compile(
    r"-?((\d+([.]\d*(e\d+)?)?)|([.]\d+([eE]\d*)?))"
)
simple_key_expr = re.compile(r"^\s*(\w+)\s*$", re.UNICODE)
simple_path_expr = re.compile(r"^\s*\w+(\s*[.]\w+\s*)+$", re.UNICODE)
compare_op_expr = re.compile(r"\s*(==|=|!=|<=|<(?!-)|>=|>)\s*")
bind_expr = re.compile(r"<([A-Za-z_]+[A-Za-z0-9_]*)>")


def lex_string(text: str, pos: int) -> tuple[str, int]:
    if pos >= len(text):
        raise ParserError("Unexpected end of path string")
    quote_char = text[pos]
    if quote_char not in "'\"":
        raise ParserError(f"Expected string at {pos} found {quote_char!r}")

    start_pos = pos
    pos += 1
    prev = pos
    output: list[str] = []
    while pos < len(text):
        char = text[pos]
        if char == quote_char:
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
            yield Key(key, start=start, end=pos)
        elif m := bind_expr.match(text, pos):
            yield Token(TKind.bind, (m.group(1),), m.group(0),
                        m.start(), m.end())
            pos = m.end()
        elif m := compare_op_expr.match(text, pos):
            op_name = m.group(1)
            op_end = m.end()
            if nm := number_expr.match(text, op_end):
                num_text = nm.group(0)
                value = float(num_text) if "." in num_text else int(num_text)
                pos = nm.end()
            else:
                value, pos = lex_string(text, op_end)
            yield Token(TKind.implicit_child, (), "", start, start)
            yield Comparison(op_name, value, start=start, end=op_end)
        else:
            for tk, expr in token_exprs.items():
                if m := expr.match(text, pos):
                    pos = m.end()
                    yield Token(tk, m.groups(), m.group(0), start, pos)
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
                        f"Expected value after {token} at {token.start}")
                return fn(reduce(left), reduce(right))
    return reducer


# The ordering in this dict defines the binding priority of the operators, from
# loosest to tightest
BinaryOpType = Callable[[JsonPath, JsonPath], JsonPath]
binary_ops: dict[TKind, BinaryOpType] = {
    TKind.choice: Choice,
    TKind.merge: Merge,
    TKind.intersect: Intersect,
    TKind.desc: Descendants,
    TKind.child: Child.make,
    TKind.implicit_child: Child.make,
    TKind.where: Where,
}
binary_op_order = tuple(binary_ops)


def reduce(tokens: list[Union[JsonPath, Token]]) -> JsonPath:
    assert tokens
    # Replace atomic tokens with their JsonPath equivalent, and also handle
    # groupings (round and square brackets)
    i = 0
    stack: list[int] = []
    while i < len(tokens):
        token = tokens[i]
        if not isinstance(token, Token):
            i += 1
            continue
        kind = token.kind
        if kind == TKind.root:
            tokens[i] = Root(start=token.start, end=token.end)
        elif kind == TKind.this:
            tokens[i] = This(start=token.start, end=token.end)
        elif kind == TKind.parent:
            tokens[i] = Parent(start=token.start, end=token.end)
        elif kind == TKind.every:
            tokens[i] = Every(start=token.start, end=token.end)
        elif kind == TKind.key:
            tokens[i] = Key(token.strings[0], start=token.start, end=token.end)
        elif kind == TKind.index:
            ix = slice_index(token.strings[0])
            tokens[i:i + 1] = [
                Token(TKind.implicit_child, (), "", token.start, token.start),
                Index(ix, start=token.start, end=token.end)
            ]
        elif kind == TKind.slice:
            start = slice_index(token.strings[0])
            end = slice_index(token.strings[1])
            step = slice_index(token.strings[3])
            tokens[i:i + 1] = [
                Token(TKind.implicit_child, (), "", token.start, token.start),
                Index(slice(start, end, step), start=token.start, end=token.end)
            ]
            i = i + 2
            continue
        elif kind in (TKind.open_paren, TKind.func,
                      TKind.open_square):
            stack.append(i)
        elif kind == TKind.close_square:
            open_index = stack.pop()
            open_square = tokens[open_index]
            if open_square.kind != TKind.open_square:
                raise ParserError(f"Unbalanced paren {token} at {token.start}")
            body = tokens[open_index + 1:i]
            if not body:
                del tokens[open_index:i + 1]
                i = open_index
                continue
            tokens[open_index:i + 1] = [
                Token(TKind.implicit_child, (), "", token.start, token.start),
                reduce(body)
            ]
            i = open_index + 2
            continue
        elif kind == TKind.close_paren:
            if not stack:
                raise ParserError(f"Unbalenced close paren at {token.start}")
            open_index = stack.pop()
            open_paren = tokens[open_index]
            if open_paren.kind not in (TKind.func, TKind.open_paren):
                raise ParserError(f"Unbalanced paren {token} at {token.start}")
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
        elif kind == TKind.bind:
            if i == 0 or not tokens:
                raise ParserError(
                    f"Binding must follow an expression at {token.start}"
                )
            name = token.strings[0]
            previous = tokens[i - 1]
            assert isinstance(previous, JsonPath)
            tokens[i - 1:i + 1] = [Bind(name, previous)]
            continue

        i += 1

    if not tokens:
        raise ParserError("String reduced to an empty path")
    tk0 = tokens[0]
    if isinstance(tokens[0], Token) and tk0.kind == TKind.implicit_child:
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
                f"Left side of {op_token} is empty at {op_token.start}")
        right = tokens[op_index + 1:]
        if not right:
            raise ParserError(
                f"Right side of {op_token} is empty at {op_token.start}")
        return operator(reduce(left), reduce(right))

    if not tokens:
        raise ParserError("String reduced to an empty path")

    if len(tokens) == 1:
        tk0 = tokens[0]
        if isinstance(tk0, JsonPath):
            return tk0
        else:
            raise ParserError(f"Parser error: {tk0} at {tk0.start}")
    else:
        tk1 = tokens[1]
        raise ParserError(f"Expected operator at {tk1.start} found {tk1}")


def _fast_keypath(p: str) -> JsonPath:
    parts = p.split(".")
    jp = Child(Key(parts[0].strip()), Key(parts[1].strip()))
    for i in range(2, len(parts)):
        jp = Child(jp, Key(parts[i].strip()))
    return jp


def parse(text: str) -> JsonPath:
    if simple_key_expr.match(text):
        return Key(text.strip())
    if simple_path_expr.match(text):
        return _fast_keypath(text)

    tokens = list(lex(text))
    if not tokens:
        raise ParserError("Parse error: no tokens")
    return reduce(tokens)
