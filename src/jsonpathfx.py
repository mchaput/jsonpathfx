from __future__ import annotations

import ast
import enum
import operator
import re
from typing import (Any, Callable, Iterable, NamedTuple, Optional, Pattern,
                    Sequence, Union)

JsonValue = Union[
    int, float, str, list["JsonValue"], tuple, dict[str, "JsonValue"], None
]
LexerFn = Callable[[str, int], Optional[tuple["Token", int]]]


class ParserError(Exception):
    pass


class Kind(enum.Enum):
    eof = enum.auto()
    root = enum.auto()  # $
    this = enum.auto()  # @
    child = enum.auto()  # .
    desc = enum.auto()  # ..
    where = enum.auto()  # <-
    star = enum.auto()  # *
    merge = enum.auto()  # |
    or_ = enum.auto()  # ||
    intersect = enum.auto()  # &
    open_paren = enum.auto()  # (
    close_paren = enum.auto()  # )
    open_square = enum.auto()  # [
    close_square = enum.auto()  # ]
    comma = enum.auto()  # ,
    colon = enum.auto()  # :
    # apply = enum.auto()  # name(
    name = enum.auto()  # name
    number = enum.auto()  # 1
    string = enum.auto()  # 'string'
    bind = enum.auto()  # <name>
    plus = enum.auto()  # +
    minus = enum.auto()  # -
    divide = enum.auto()  # /
    neg = enum.auto()  # !
    less_than = enum.auto()  # <
    less_than_eq = enum.auto()  # <=
    equals = enum.auto()  # ==
    greater_than = enum.auto()  # >
    greater_than_eq = enum.auto()  # >=
    not_eq = enum.auto()  # !=
    regex = enum.auto()  # ~=


class Precedence(enum.IntEnum):
    where = 1
    comparison = 2
    binary = 3
    intersect = 4
    merge = 5
    or_ = 6
    sum = 7
    product = 8
    child = 11
    postfix = 12
    bind = 20
    call = 100


class Token(NamedTuple):
    kind: Kind
    payload: Any
    pos: int


ws_expr = re.compile(r"\s+")
# slice_expr = re.compile(r"(\s*-?\d+\s*)?:(\s*-?\d+\s*)?(:-?\d+)?")


def get_keys(value: JsonValue) -> JsonValue:
    if isinstance(value, dict):
        return list(value)
    else:
        return []


def regex_compare(expr: JsonValue, target: JsonValue) -> bool:
    if isinstance(expr, str) and isinstance(target, str):
        return bool(re.search(expr, target))
    return False


def lex_string_literal(text: str, pos: int) -> Optional[tuple[Token, int]]:
    quote_char = text[pos]
    if quote_char not in "'\"":
        return

    start_pos = pos
    pos += 1
    prev = pos
    string = ""
    while pos < len(text):
        char = text[pos]
        if char == quote_char:
            if prev < pos:
                string += text[prev:pos]
            return Token(Kind.string, string, start_pos), pos + 1
        elif text[pos] == "\\" and pos < len(text) - 1:
            if prev < pos:
                string += text[prev:pos]
            string += text[pos + 1]
            pos += 2
            prev = pos
        else:
            pos += 1


# def lex_slice(text: str, pos: int) -> Optional[tuple[Token, int]]:
#     if m := slice_expr.match(text, pos):
#         start = int(m.group(1))
#         stop = int(m.group(2)) if m.group(2) else None
#         step = int(m.group(3)[1:]) if m.group(3) else None
#         return Token(Kind.slice, (start, stop, step), pos), m.end()


token_exprs: dict[Kind, Union[str, Pattern, LexerFn]] = {
    # The order is significant! For strings that share a prefix, the longer
    # should come first
    Kind.root: "$",
    Kind.this: "@",
    Kind.desc: "..",
    Kind.child: ".",
    Kind.where: "<-",
    Kind.star: "*",
    Kind.or_: "||",
    Kind.merge: "|",
    Kind.intersect: "&",
    Kind.open_paren: "(",
    Kind.close_paren: ")",
    Kind.open_square: "[",
    Kind.close_square: "]",
    Kind.comma: ",",
    Kind.colon: ":",
    # Kind.apply: re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*[(]")
    Kind.number: re.compile(r"(-?\d+([.]\d*(e\d+)?)?)|([.]\d+([eE]\d*)?)"),
    Kind.bind: re.compile(r"(\w+):", re.UNICODE),
    Kind.name: re.compile(r"(\w+)", re.UNICODE),
    Kind.string: lex_string_literal,
    # Kind.slice: lex_slice,
    Kind.plus: "+",
    Kind.minus: "-",
    Kind.divide: "/",
    Kind.less_than_eq: "<=",
    Kind.less_than: "<",
    Kind.equals: re.compile("(==?)"),
    Kind.greater_than_eq: ">=",
    Kind.greater_than: ">",
    Kind.not_eq: "!=",
    Kind.neg: "!",
    Kind.regex: "~="
}


def lex_token(text: str, pos: int) -> Optional[tuple[Token, int]]:
    for kind, expr in token_exprs.items():
        if isinstance(expr, str):
            if text.startswith(expr, pos):
                return Token(kind, expr, pos), pos + len(expr)
        elif callable(expr):
            result = expr(text, pos)
            if result:
                return result
        elif m := expr.match(text, pos):
            return Token(kind, m.group(1), pos), m.end()


def lex(text: str) -> Sequence[Token]:
    pos = 0
    tokens: list[Token] = []
    while pos < len(text):
        if m := ws_expr.match(text, pos):
            pos = m.end()
            if pos == len(text):
                break

        result = lex_token(text, pos)
        if not result:
            raise ParserError(f"Can't parse {text[pos]!r} at {pos}")
        token, new_pos = result
        if new_pos <= pos:
            raise ParserError(f"Parsing went backwards at {pos}: "
                              f"{token} -> {new_pos}")
        pos = new_pos
        tokens.append(token)
    return tokens


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

    def push(self, key: str | int | None, value: JsonValue, name: str = None
             ) -> Match:
        return Match(value, key, self.parents + (self,), name)

    def path(self) -> Sequence[str | int | None]:
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
    def __init__(self):
        self._pos = -1

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

    def pos(self) -> int:
        return self._pos

    def set_pos(self, pos: int) -> None:
        self._pos = pos


class BinaryJsonPath(JsonPath):
    # Abstract base path for binary path operators

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

    def pos(self) -> int:
        return self.left.pos()

    def set_pos(self, pos: int) -> None:
        self.left.set_pos(pos)


class UnaryJsonPath(JsonPath):
    # Abstract base path for unary path operators

    def __init__(self, child: JsonPath):
        super().__init__()
        self.child = child

    def __repr__(self):
        return f"{type(self).__name__}({self.child!r})"

    def pos(self) -> int:
        return self.child.pos()

    def set_pos(self, pos: int) -> None:
        self.child.pos = pos


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


class Literal(JsonPath):
    def __init__(self, literal: Union[int, float, str]):
        super().__init__()
        self.literal = literal

    def __hash__(self):
        return hash(type(self)) ^ hash(self.literal)

    def __eq__(self, other: Literal):
        return type(self) is type(other) and self.literal == other.literal

    def __repr__(self):
        return f"{type(self).__name__}({self.literal!r})"

    def find(self, match: Union[JsonValue, Match]) -> Iterable[Match]:
        yield match.push(self.literal, self.literal)


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


class Or(BinaryJsonPath):
    def find(self, match: Union[JsonValue, Match]) -> Iterable[Match]:
        match = ensure(match)
        left_matched = False
        for match in self.left.find(match):
            yield match
            left_matched = True
        if not left_matched:
            yield from self.right.find(match)


# Can't call this Union, that's a type annotation
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
    def __init__(self, key: str):
        super().__init__()
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
    def __init__(self, ix: Union[int, slice]):
        super().__init__()
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


class Bind(UnaryJsonPath):
    def __init__(self, child: JsonPath, name: str):
        super().__init__(child)
        self.name = name

    def __repr__(self):
        return f"{type(self).__name__}({self.name!r}, {self.child!r})"

    def __eq__(self, other):
        return (type(self) is type(other) and
                self.name == other.name and
                self.child == other.child)

    def __hash__(self):
        return hash(type(self)) ^ hash(self.name)

    def find(self, match: Union[JsonValue, Match]) -> Iterable[Match]:
        match = ensure(match)
        for found in self.child.find(match):
            yield match.push(found.key, found.value, self.name)


class FilterFunction(JsonPath):
    def __init__(self, fn: Callable[[JsonValue], JsonValue], *args: JsonPath):
        super().__init__()
        self.fn = fn
        self.args = args

    @classmethod
    def wraps(cls, fn: Callable[[JsonValue], JsonValue]
              ) -> Callable[[Sequence[JsonPath]], JsonPath]:
        def wrap_filter_fn(*args: Sequence[JsonPath]) -> JsonPath:
            return FilterFunction(fn, *args)
        return wrap_filter_fn

    def __repr__(self):
        return f"{type(self).__name__}({self.fn!r})"

    def __eq__(self, other):
        return type(self) is type(other) and self.fn == other.fn

    def __hash__(self):
        return hash(type(self)) ^ hash(self.fn)

    def find(self, match: Union[JsonValue, Match]) -> Iterable[Match]:
        match = ensure(match)
        value = self.fn(match.value)
        yield match.push(value, value)


class Comparison(BinaryJsonPath):
    # Allow creating an instance with a string instead of a function, to make
    # it easier to compare in tests
    string_to_op: dict[str, Callable] = {
        "!=": operator.ne,
        "==": operator.eq,
        "<=": operator.le,
        "<": operator.lt,
        ">=": operator.ge,
        ">": operator.gt,
        "~=": regex_compare,
    }

    def __init__(self, left: JsonPath, op: Union[str, Callable],
                 right: JsonPath):
        super().__init__(left, right)
        if isinstance(op, str):
            op = self.string_to_op[op]
        self.op = op

    def __repr__(self):
        return f"{type(self).__name__}({self.left!r} {self.op} {self.right!r})"

    def __hash__(self):
        return hash(type(self)) ^ hash(self.op) ^ hash(self.val)

    def __eq__(self, other):
        return (type(self) is type(other) and
                self.left == other.left and
                self.op == other.op and
                self.right == other.right)

    def find(self, match: Union[JsonValue, Match]) -> Iterable[Match]:
        match = ensure(match)
        op = self.op

        # Evaluate the right side once to get the value we will compare to
        for right_m in self.right.find(match):
            right_v = right_m.value
            if isinstance(right_v, (int, float, str)):
                break
        else:
            return

        for left_m in self.left.find(match):
            try:
                passed = op(left_m.value, right_v)
            except TypeError:
                continue
            if passed:
                yield left_m


class Math(BinaryJsonPath):
    # Allow creating an instance with a string instead of a function, to make
    # it easier to compare in tests
    string_to_op: dict[str, Callable] = {
        "+": operator.add,
        "-": operator.sub,
        "*": operator.mul,
        "/": operator.truediv,
    }

    def __init__(self, left: JsonPath, op: Union[str, Callable],
                 right: JsonPath):
        super().__init__(left, right)
        if isinstance(op, str):
            op = self.string_to_op[op]
        self.op = op

    def find(self, match: Union[JsonValue, Match]) -> Iterable[Match]:
        match = ensure(match)
        op = self.op

        # Evaluate the right side once to get the value we will compare to
        for right_m in self.right.find(match):
            right_v = right_m.value
            if isinstance(right_v, (int, float, str)):
                break
        else:
            return

        for left_m in self.left.find(match):
            left_v = left_m.value
            if isinstance(left_v, (int, float)):
                result = op(left_v, right_v)
                yield match.push(left_v, result)


filter_functions: dict[str, Callable[[], JsonPath]] = {
    "len": FilterFunction.wraps(len),
    "sorted": FilterFunction.wraps(sorted),
    "keys": FilterFunction.wraps(get_keys),
    "parent": Parent,
}


# Pratt parser based on this extremely helpful and easy-to-read article:
# https://journal.stuffwithstuff.com/2011/03/19/pratt-parsers-expression-parsing-made-easy/
#
# In the class names below "prefix" means "prefix or standalone" and
# "infix" means "not prefix" (includes postfix and "mixfix" (e.g. ?:))

class Parselet:
    precedence = Precedence.binary

    def parse_prefix(self, parser: Parser, token: Token) -> JsonPath:
        raise NotImplementedError

    def parse_infix(self, parser: Parser, left: JsonPath, token: Token
              ) -> JsonPath:
        raise NotImplementedError


class GroupParselet(Parselet):
    def parse_prefix(self, parser: Parser, token: Token) -> JsonPath:
        expr = parser.expression()
        parser.consume(Kind.close_paren)
        return expr


class SingletonParselet(Parselet):
    def __init__(self, path: JsonPath):
        self.path = path

    def __repr__(self):
        return f"<{type(self).__name__} {self.path!r}>"

    def parse_prefix(self, parser: Parser, token: Token) -> JsonPath:
        return self.path


class KeyParselet(Parselet):
    def parse_prefix(self, parser: Parser, token: Token) -> JsonPath:
        return Key(token.payload)


class LiteralParselet(Parselet):
    def parse_prefix(self, parser: Parser, token: Token) -> JsonPath:
        if token.kind == Kind.string:
            return Literal(token.payload)
        elif token.kind == Kind.number:
            num_str = token.payload
            try:
                number = ast.literal_eval(num_str)
            except SyntaxError:
                raise ParserError(f"Can't parse number {num_str}")
            return Literal(number)
        else:
            raise Exception(f"Unknown literal kind {token.kind}")


class IndexParselet(Parselet):
    # This class can act as both a prefix, for `[foo]`, and as an infix to
    # support `foo[bar]` without a dot in-between

    def parse_prefix(self, parser: Parser, token: Token) -> JsonPath:
        return self.parse_numeric_index(parser)

    def parse_infix(self, parser: Parser, left: JsonPath, token: Token
                    ) -> JsonPath:
        right = self.parse_numeric_index(parser)
        return Child.make(left, right)

    @staticmethod
    def to_index(string: str) -> int:
        try:
            return int(string)
        except ValueError:
            raise ParserError(f"Can't use {string} as index")

    @staticmethod
    def parse_numeric_index(parser: Parser) -> JsonPath:
        # If square brackets contains a number or slice syntax, parse it into
        # an Index path, otherwise, treat x[y] just like x.(y)
        if parser.current().kind in (Kind.number, Kind.colon):
            to_index = IndexParselet.to_index
            start_str = ""
            stop_str = ""
            step_str = ""
            if parser.current().kind == Kind.number:
                start_str = parser.consume().payload
                if parser.take(Kind.close_square):
                    return Index(to_index(start_str))

            if parser.take(Kind.colon):
                if parser.current().kind == Kind.number:
                    stop_str = parser.consume().payload
                if parser.take(Kind.colon):
                    if parser.current().kind == Kind.number:
                        step_str = parser.consume().payload
            parser.consume(Kind.close_square)
            start = to_index(start_str) if start_str else None
            stop = to_index(stop_str) if stop_str else None
            step = to_index(step_str) if step_str else None
            return Index(slice(start, stop, step))
        else:
            expr = parser.expression()
            parser.consume(Kind.close_square)
            return expr


class BindParselet(Parselet):
    def parse_prefix(self, parser: Parser, token: Token) -> JsonPath:
        name: str = token.payload
        child = parser.expression(Precedence.bind)
        return Bind(child, name)


class CombiningParselet(Parselet):
    def __init__(self, path_maker: Callable[[JsonPath, JsonPath], JsonPath],
                 prcendence=Precedence.binary, right_assoc=False):
        self.path_maker = path_maker
        self.precedence = prcendence
        self.right_assoc = right_assoc

    def __repr__(self):
        return f"<{type(self).__name__} {self.path_maker}, {self.precedence}>"

    def parse_infix(self, parser: Parser, left: JsonPath, token: Token
                    ) -> JsonPath:
        prec_adjust = -1 if self.right_assoc else 0
        right = parser.expression(self.precedence + prec_adjust)
        return self.path_maker(left, right)


class AbstractOperatorParselet(Parselet):
    def __init__(self, op: Union[str, Callable], precedence=Precedence.binary,
                 right_assoc=False):
        self.op = op
        self.precedence = precedence
        self.right_assoc = right_assoc


class ComparisonParselet(AbstractOperatorParselet):
    def parse_infix(self, parser: Parser, left: JsonPath, token: Token
                    ) -> JsonPath:
        prec_adjust = -1 if self.right_assoc else 0
        # Hack: we want to be able to compare to a string, but normally "foo"
        # would parse to Key("foo"), so we special case it
        if parser.current().kind == Kind.string:
            right = Literal(parser.consume().payload)
        else:
            right = parser.expression(self.precedence + prec_adjust)
        return Comparison(left, self.op, right)


class MathParselet(AbstractOperatorParselet):
    def parse_infix(self, parser: Parser, left: JsonPath, token: Token
                    ) -> JsonPath:
        prec_adjust = -1 if self.right_assoc else 0
        right = parser.expression(self.precedence + prec_adjust)
        return Math(left, self.op, right)


class CallParselet(Parselet):
    precedence = Precedence.call

    def parse_infix(self, parser: Parser, left: JsonPath, token: Token
              ) -> JsonPath:
        if not isinstance(left, Key):
            raise ParserError(f"Unexpected left bracket at {token.pos}")
        fn_name = left.key

        args: list[JsonPath] = []
        if not parser.take(Kind.close_paren):
            # Parse comma-separated arguments until we hit close paren
            while True:
                args.append(parser.expression())
                if parser.take(Kind.comma):
                    continue
                else:
                    break
            parser.consume(Kind.close_paren)

        try:
            maker_fn = filter_functions[fn_name]
        except KeyError:
            raise ParserError(f"No function named {fn_name}")
        return maker_fn(*args)


prefixes: dict[Kind, Parselet] = {
    Kind.number: LiteralParselet(),
    Kind.name: KeyParselet(),
    Kind.string: KeyParselet(),
    Kind.root: SingletonParselet(Root()),
    Kind.this: SingletonParselet(This()),
    Kind.star: SingletonParselet(Every()),
    Kind.bind: BindParselet(),
    Kind.open_paren: GroupParselet(),
    Kind.open_square: IndexParselet(),
}
infixes: dict[Kind, Parselet] = {
    Kind.child: CombiningParselet(Child.make, Precedence.child),
    Kind.where: CombiningParselet(Where, Precedence.where),
    Kind.desc: CombiningParselet(Descendants, Precedence.binary),
    Kind.merge: CombiningParselet(Merge, Precedence.merge),
    Kind.or_: CombiningParselet(Or, Precedence.or_),
    Kind.intersect: CombiningParselet(Intersect, Precedence.intersect),
    Kind.less_than_eq: ComparisonParselet(operator.le),
    Kind.less_than: ComparisonParselet(operator.lt),
    Kind.equals: ComparisonParselet(operator.eq),
    Kind.greater_than_eq: ComparisonParselet(operator.ge),
    Kind.greater_than: ComparisonParselet(operator.gt),
    Kind.not_eq: ComparisonParselet(operator.ne),
    Kind.plus: MathParselet(operator.add, Precedence.sum),
    Kind.minus: MathParselet(operator.sub, Precedence.sum),
    Kind.star: MathParselet(operator.mul, Precedence.product),
    Kind.divide: MathParselet(operator.truediv, Precedence.product),
    Kind.open_paren: CallParselet(),
    Kind.open_square: IndexParselet(),
}


class Parser:
    def __init__(self, tokens: Sequence[Token]):
        self.tokens = list(tokens)
        # self.depth = 0

    def take(self, kind: Kind) -> Optional[Token]:
        if not self.tokens:
            return None
        if self.tokens[0].kind == kind:
            return self.consume()

    def consume(self, kind: Kind = None) -> Token:
        if self.tokens:
            token = self.tokens.pop(0)
        else:
            token = Token(Kind.eof, "<EOF>", -1)

        if kind and token.kind != kind:
            raise ParserError(f"Expected {kind}, "
                              f"found {token.kind} ({token.payload}) "
                              f"at {token.pos}")
        return token

    def current(self) -> Token:
        return self.lookahead(0)

    def lookahead(self, distance: int) -> Optional[Token]:
        if distance >= len(self.tokens):
            return Token(Kind.eof, "<EOF>", -1)
        return self.tokens[distance]

    def current_infix(self) -> Optional[Parselet]:
        return infixes.get(self.current().kind)

    def infix_precedence(self) -> int:
        infix = infixes.get(self.current().kind)
        if infix:
            return infix.precedence
        return 0

    def expression(self, precedence=0) -> JsonPath:
        token = self.consume()
        prefix_parselet = prefixes.get(token.kind)
        if not prefix_parselet:
            raise ParserError(f"Syntax error {token.payload} at {token.pos}")

        # self.depth += 1
        expr = prefix_parselet.parse_prefix(self, token)
        # self.depth -= 1

        while precedence < self.infix_precedence():
            token = self.consume()
            infix_parselet = infixes[token.kind]
            # self.depth += 1
            expr = infix_parselet.parse_infix(self, expr, token)
            # self.depth -= 1

        return expr

    @classmethod
    def parse(cls, text: str) -> JsonPath:
        parser = cls(lex(text))
        path = parser.expression()
        if parser.tokens:
            tk = parser.tokens[0]
            raise ParserError(f"Syntax error: {tk.payload} at {tk.pos}")
        return path


def _fast_keypath(p: str) -> JsonPath:
    parts = p.split(".")
    jp = Child(Key(parts[0].strip()), Key(parts[1].strip()))
    for i in range(2, len(parts)):
        jp = Child(jp, Key(parts[i].strip()))
    return jp


simple_key_expr = re.compile(r"^\s*(\w+)\s*$", re.UNICODE)
simple_path_expr = re.compile(r"^\s*\w+(\s*[.]\w+\s*)+$", re.UNICODE)


def parse(text: str) -> JsonPath:
    if simple_key_expr.match(text):
        return Key(text.strip())
    if simple_path_expr.match(text):
        return _fast_keypath(text)
    return Parser.parse(text)

