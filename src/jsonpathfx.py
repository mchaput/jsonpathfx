# BSD 2-Clause License
#
# Copyright (c) 2023, Matt Chaput
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import annotations
import ast
import enum
import operator
import re
import sys
from typing import (Any, Callable, Iterable, NamedTuple, Optional, Pattern,
                    Sequence, Union)

__all__ = (
    "JsonValue", "JsonPath", "ParserError", "BinaryJsonPath", "UnaryJsonPath",
    "Root", "This", "Parent", "Every", "Literal", "Child", "Where",
    "Descendants", "Or", "Merge", "Intersect", "Discard", "Key", "Index",
    "Bind", "TransformFunction", "LookupComputedKey", "Comparison", "Math",
    "parse",
)


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
    star = enum.auto()  # *
    merge = enum.auto()  # |
    and_ = enum.auto()  # &&
    or_ = enum.auto()  # ||
    intersect = enum.auto()  # &
    bang = enum.auto()  # !
    open_paren = enum.auto()  # (
    close_paren = enum.auto()  # )
    open_square = enum.auto()  # [
    close_square = enum.auto()  # ]
    open_brace = enum.auto()  # {
    close_brace = enum.auto()  # }
    comma = enum.auto()  # ,
    colon = enum.auto()  # :
    name = enum.auto()  # name
    number = enum.auto()  # 1
    string = enum.auto()  # 'string'
    bind = enum.auto()  # <name>
    plus = enum.auto()  # +
    minus = enum.auto()  # -
    divide = enum.auto()  # /
    less_than = enum.auto()  # <
    less_than_eq = enum.auto()  # <=
    equals = enum.auto()  # ==
    greater_than = enum.auto()  # >
    greater_than_eq = enum.auto()  # >=
    not_eq = enum.auto()  # !=
    regex = enum.auto()  # ~=
    comment = enum.auto()  # #...


class Precedence(enum.IntEnum):
    where = 1
    binary = 20
    intersect = 21
    merge = 22
    logic = 23
    comparison = 30
    sum = 40
    product = 41
    child = 50
    postfix = 80
    bind = 90
    call = 100


class Token(NamedTuple):
    kind: Kind
    payload: Any
    pos: int


ws_expr = re.compile(r"\s+")
# slice_expr = re.compile(r"(\s*-?\d+\s*)?:(\s*-?\d+\s*)?(:-?\d+)?")


def hashable(value: JsonValue) -> Union[int, float, str, tuple]:
    if isinstance(value, list):
        return tuple(hashable(x) for x in value)
    elif isinstance(value, dict):
        return tuple((k, hashable(v)) for k, v in value.items())
    else:
        return value


def to_index(string: str) -> int:
    try:
        return int(string)
    except ValueError:
        raise ParserError(f"Can't use {string} as index")


def get_keys(value: JsonValue) -> JsonValue:
    if isinstance(value, dict):
        return list(value)
    else:
        return []


def get_items(value: JsonValue) -> JsonValue:
    if isinstance(value, dict):
        # Change tuples into lists so they're proper JSON values
        return [list(item) for item in value.items()]
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


token_exprs: dict[Kind, Union[str, Pattern, LexerFn]] = {
    # The order is significant! For strings that share a prefix, the longer
    # should come first
    Kind.root: "$",
    Kind.this: "@",
    Kind.desc: "..",
    Kind.child: ".",
    Kind.star: "*",
    Kind.or_: "||",
    Kind.and_: "&&",
    Kind.merge: "|",
    Kind.intersect: "&",
    Kind.open_paren: "(",
    Kind.close_paren: ")",
    Kind.open_square: "[",
    Kind.close_square: "]",
    Kind.open_brace: "{",
    Kind.close_brace: "}",
    Kind.comma: ",",
    Kind.colon: ":",
    Kind.number: re.compile(r"(-?\d+([.]\d*(e\d+)?)?)|([.]\d+([eE]\d*)?)"),
    Kind.bind: re.compile(r"(\w+):", re.UNICODE),
    Kind.name: re.compile(r"(\w+)", re.UNICODE),
    Kind.string: lex_string_literal,
    Kind.plus: "+",
    Kind.minus: "-",
    Kind.divide: "/",
    Kind.less_than_eq: "<=",
    Kind.less_than: "<",
    Kind.equals: re.compile("(==?)"),
    Kind.greater_than_eq: ">=",
    Kind.greater_than: ">",
    Kind.not_eq: "!=",
    Kind.regex: "~=",
    Kind.bang: "!",
    Kind.comment: re.compile("#([^\n]*)")
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


def lex(text: str) -> list[Token]:
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

        if token.kind != Kind.comment:
            tokens.append(token)
    return tokens


class Match(NamedTuple):
    value: JsonValue
    key: str | int | None
    parents: tuple[Match, ...]
    name: Optional[str] = None
    debug: bool = False
    debug_indent: int = 0

    @property
    def root(self) -> Match:
        if self.parents:
            return self.parents[0]
        else:
            return self

    def push_parent(self, key: str | int | None,
                    value: JsonValue, name: str = None) -> Match:
        if not isinstance(key, (int, float, str)) and key is not None:
            raise TypeError(f"Can't use {type(key)} as key")
        if not isinstance(value, (int, float, str, dict, list, tuple)):
            raise TypeError(f"Can't use {type(value)} as value")
        return Match(value, key, self.parents + (self,), name, self.debug,
                     self.debug_indent + 1)

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

    def plus_level(self) -> Match:
        if self.debug:
            return Match(self.value, self.key, self.parents, self.name,
                         self.debug, self.debug_indent + 1)
        else:
            return self


def every_child(obj: JsonPath, match: Match) -> Iterable[Match]:
    v = match.value
    if isinstance(v, dict):
        for key, value in v.items():
            if match.debug:
                debug_msg(obj, match, f"found dict item {key!r}")
            yield match.push_parent(key, value)
    else:
        try:
            iterator = iter(v)
        except TypeError:
            if match.debug:
                debug_msg(obj, match, "Can't interate on {v!r}")
        else:
            for i, subval in enumerate(iterator):
                if match.debug:
                    debug_msg(obj, match, f"Found list item {subval!r}")
                yield match.push_parent(i, subval)


def debug_msg(obj: JsonPath, match: Match, msg: str) -> None:
    print("  " * match.debug_indent, end="", file=sys.stderr)
    print(f"{type(obj).__name__}:", msg, file=sys.stderr)
    sys.stderr.flush()


class JsonPath:
    def __init__(self):
        self._pos = -1

    def __repr__(self):
        return f"{type(self).__name__}()"

    def __eq__(self, other):
        # Good enough for "singletons"; classes with parameters should override
        return type(self) is type(other)

    def __hash__(self):
        # Good enough for "singletons"; classes with parameters should override
        return hash(type(self))

    def find(self, match: Union[JsonValue, Match], debug=False
             ) -> Iterable[Match]:
        if isinstance(match, Match):
            match = match.plus_level()
        else:
            match = Match(match, None, (), None, debug)
        return self._find(match)

    def _find(self, match: Match) -> Iterable[Match]:
        raise NotImplementedError

    def values(self, data: JsonValue, debug=False) -> Sequence[JsonValue]:
        return list(self.itervalues(data, debug=debug))

    def itervalues(self, data: JsonValue, debug=False) -> Iterable[JsonValue]:
        return (match.value for match in self.find(data, debug=debug))

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

    def __hash__(self):
        return hash(type(self)) ^ hash(self.child)

    def __eq__(self, other):
        return type(self) is type(other) and self.child == other.child

    def __repr__(self):
        return f"{type(self).__name__}({self.child!r})"

    def pos(self) -> int:
        return self.child.pos()

    def set_pos(self, pos: int) -> None:
        self.child.pos = pos


class Root(JsonPath):
    def _find(self, match: Match) -> Iterable[Match]:
        if match.debug:
            debug_msg(self, match, "found root node")
        yield match.root


class This(JsonPath):
    def _find(self, match: Match) -> Iterable[Match]:
        if match.debug:
            debug_msg(self, match, "found: {match.value")
        yield match.push_parent(None, match.value)


class Parent(JsonPath):
    def _find(self, match: Match) -> Iterable[Match]:
        if match.parents:
            if match.debug:
                debug_msg(self, match, "found: {match.parents[-1]")
            yield match.parents[-1]
        elif match.debug:
            debug_msg(self, match, "!no parent")


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

    def _find(self, match: Match) -> Iterable[Match]:
        if match.debug:
            debug_msg(self, match, f"found literal {self.literal}")
        yield match.push_parent(self.literal, self.literal)


class Every(JsonPath):
    def _find(self, match: Match) -> Iterable[Match]:
        return every_child(self, match)


class Child(BinaryJsonPath):
    @classmethod
    def make(cls, left: JsonPath, right: JsonPath) -> JsonPath:
        if isinstance(left, This):
            return right
        elif isinstance(right, This):
            return left
        elif isinstance(right, Root):
            return right
        else:
            return Child(left, right)

    def _find(self, match: Match) -> Iterable[Match]:
        for left_m in self.left.find(match):
            for right_m in self.right.find(left_m):
                if match.debug:
                    debug_msg(self, match, f"found {right_m.value!r}")
                yield right_m


class Descendants(BinaryJsonPath):
    def _find(self, match: Match) -> Iterable[Match]:
        def match_recursively(this: Match) -> Iterable[Match]:
            for m in self.right.find(this):
                yield m

            val = this.value
            if isinstance(val, dict):
                for key, subval in val.items():
                    yield from match_recursively(this.push_parent(key, subval))
            elif isinstance(val, (list, tuple)):
                for i, subval in enumerate(val):
                    yield from match_recursively(this.push_parent(i, subval))

        for left_match in self.left.find(match):
            yield from match_recursively(left_match)


class Or(BinaryJsonPath):
    def _find(self, match: Match) -> Iterable[Match]:
        left_matched = False
        for left_m in self.left.find(match):
            if match.debug and not left_matched:
                debug_msg(self, match, f"left side matched")
            left_matched = True
            yield left_m
        if not left_matched:
            if match.debug:
                debug_msg(self, match, "!no match on left")
            yield from self.right.find(match)


class And(BinaryJsonPath):
    def _find(self, match: Match) -> Iterable[Match]:
        left_matched = False
        for left_m in self.left.find(match):
            if match.debug:
                debug_msg(self, match, f"left matched: {left_m.value}")
            for right_m in self.right.find(match):
                if match.debug:
                    debug_msg(self, match, f"right matched: {right_m.value}")
                yield right_m
                break


# Can't call this Union, that's a type annotation
class Merge(BinaryJsonPath):
    def _find(self, match: Match) -> Iterable[Match]:
        seen: set[int] = set()
        for subm in self.left.find(match):
            if match.debug:
                debug_msg(self, match, f"left found {subm.value!r}")
            yield subm
            seen.add(id(subm.value))
        for subm in self.right.find(match):
            if id(subm.value) not in seen:
                if match.debug:
                    debug_msg(self, match, f"right found {subm.value!r}")
                yield subm


class Intersect(BinaryJsonPath):
    def _find(self, match: Match) -> Iterable[Match]:
        seen: set[Union[int, float, str]] = set()
        for left_m in self.left.find(match):
            seen.add(hashable(left_m.value))
            if match.debug:
                debug_msg(self, match, f"left: {left_m.value!r}")
        for right_m in self.right.find(match):
            if hashable(right_m.value) in seen:
                if match.debug:
                    debug_msg(self, match, f"found {right_m.value!r}")
                yield right_m
            elif match.debug:
                debug_msg(self, match, f"!no match {right_m.value}")


class Discard(BinaryJsonPath):
    def _find(self, match: Match) -> Iterable[Match]:
        for left_m in self.left.find(match):
            passed = True
            for right_m in self.right.find(left_m):
                if match.debug:
                    debug_msg(self, match, f"!filtered out {right_m.value!r}")
                passed = False
                break
            if passed:
                if match.debug:
                    debug_msg(self, match, f"found {left_m.value!r}")
                yield left_m


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

    def _find(self, match: Match) -> Iterable[Match]:
        this = match.value
        if isinstance(this, dict):
            try:
                value = this[self.key]
            except KeyError:
                if match.debug:
                    debug_msg(self, match, "!key not found: {key!r}")
            else:
                if match.debug:
                    debug_msg(self, match, f"{self.key!r} = {value!r}")
                yield match.push_parent(self.key, value)
        elif match.debug:
            debug_msg(self, match, f"!not a dict: {this!r}")


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

    def _find(self, match: Match) -> Iterable[Match]:
        this = match.value
        if isinstance(this, (list, tuple)):
            ix = self.index
            if isinstance(ix, int):
                try:
                    yield match.push_parent(ix, this[ix])
                except IndexError:
                    if match.debug:
                        debug_msg(self, match, f"index not found: {ix!r}")
                else:
                    if match.debug:
                        debug_msg(self, match, f"found {this[ix]!r}")
            elif isinstance(ix, slice):
                start, stop, step = ix.indices(len(this))
                for i in range(start, stop, step):
                    if match.debug:
                        debug_msg(self, match, f"found {this[i]}")
                    yield match.push_parent(i, this[i])


class Where(UnaryJsonPath):
    def _find(self, match: Match) -> Iterable[Match]:
        for subm in self.child.find(match):
            if match.debug:
                debug_msg(self, match, f"found {subm.value!r}")
            yield match
            break


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

    def _find(self, match: Match) -> Iterable[Match]:
        for subm in self.child.find(match):
            if match.debug:
                debug_msg(self, match, f"found {subm.value!r}")
            yield match.push_parent(subm.key, subm.value, self.name)


class TransformFunction(JsonPath):
    def __init__(self, fn: Callable[[JsonValue], JsonValue], *args: JsonPath,
                 unwrap=False):
        super().__init__()
        self.fn = fn
        self.args = args
        self.unwrap = unwrap

    @classmethod
    def for_function(cls, fn: Callable[[JsonValue], JsonValue],
                     unwrap=False) -> Callable[[Sequence[JsonPath]], JsonPath]:
        def wrap_filter_fn(*args: Sequence[JsonPath]) -> JsonPath:
            return TransformFunction(fn, *args, unwrap=unwrap)
        return wrap_filter_fn

    def __repr__(self):
        return f"{type(self).__name__}({self.fn!r})"

    def __eq__(self, other):
        return type(self) is type(other) and self.fn == other.fn

    def __hash__(self):
        return hash(type(self)) ^ hash(self.fn)

    def _find(self, match: Match) -> Iterable[Match]:
        value = self.fn(match.value)
        if match.debug:
            debug_msg(self, match, f"fn returned {value!r}")
        if self.unwrap:
            for subval in value:
                yield match.push_parent(None, subval)
        else:
            yield match.push_parent(None, value)


class LookupComputedKey(UnaryJsonPath):
    def _find(self, match: Match) -> Iterable[Match]:
        this = match.value
        if not isinstance(this, dict):
            return

        for subm in self.child.find(match):
            key = subm.value
            if isinstance(key, str):
                yield match.push_parent(key, this[key])
                return


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
        return (hash(type(self)) ^ hash(self.left) ^ hash(self.op) ^
                hash(self.right))

    def __eq__(self, other):
        return (type(self) is type(other) and
                self.left == other.left and
                self.op == other.op and
                self.right == other.right)

    def _find(self, match: Match) -> Iterable[Match]:
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
            try:
                passed = op(left_v, right_v)
            except TypeError:
                if match.debug:
                    debug_msg(self, match, f"can't compare {left_v}")
                continue
            if passed:
                if match.debug:
                    debug_msg(self, match,
                                f"found {left_v} {self.op} {right_v}")
                yield left_m
            elif match.debug:
                debug_msg(self, match,
                            f"!False: {left_v} {self.op} {right_v}")


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

    def _find(self, match: Match) -> Iterable[Match]:
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
            if match.debug:
                debug_msg(self, match, f"left value {left_v}")

            if isinstance(left_v, (int, float)):
                result = op(left_v, right_v)
                if match.debug:
                    debug_msg(self, match, f"result {result!r}")
                yield match.push_parent(left_v, result)


transform_functions: dict[str, Callable[[], JsonPath]] = {
    "len": TransformFunction.for_function(len),
    "sorted": TransformFunction.for_function(sorted),
    "keys": TransformFunction.for_function(get_keys),
    "items": TransformFunction.for_function(get_items, unwrap=True),
    "lookup": LookupComputedKey,
    "parent": Parent,
}


# Pratt parser based on this extremely helpful and easy-to-read article:
# https://journal.stuffwithstuff.com/2011/03/19/pratt-parsers-expression-parsing-made-easy/
#
# In the class names below "prefix" means "prefix or standalone" and
# "infix" means "not prefix" (includes postfix and "mixfix" (e.g. ?:))

class Parselet:
    def parse_prefix(self, parser: Parser, token: Token) -> JsonPath:
        raise NotImplementedError

    def parse_infix(self, parser: Parser, left: JsonPath, token: Token
                    ) -> JsonPath:
        raise NotImplementedError

    def precedence(self) -> int:
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


class ImplictChildBaseParselet(Parselet):
    # Subclasses can act as both a prefix, e.g. `[foo]`, and as an infix to
    # support e.g. `foo[bar]` without a dot in-between
    def parse_infix(self, parser: Parser, left: JsonPath, token: Token
                    ) -> JsonPath:
        right = self.parse_prefix(parser, token)
        return Child.make(left, right)

    def precedence(self) -> int:
        return Precedence.child


class IndexParselet(ImplictChildBaseParselet):
    def parse_prefix(self, parser: Parser, token: Token) -> JsonPath:
        # If square brackets contains a number or slice syntax, parse it into
        # an Index path, otherwise, treat x[y] just like x.(y)
        if parser.current().kind in (Kind.number, Kind.colon):
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


class WhereParselet(ImplictChildBaseParselet):
    def parse_prefix(self, parser: Parser, token: Token) -> JsonPath:
        expr = parser.expression()
        parser.consume(Kind.close_brace)
        return Where(expr)


class BindParselet(Parselet):
    def parse_prefix(self, parser: Parser, token: Token) -> JsonPath:
        name: str = token.payload
        child = parser.expression(self.precedence())
        return Bind(child, name)

    def precedence(self) -> int:
        return Precedence.bind


class UnaryParslet(Parselet):
    def __init__(self, path_type: type[UnaryJsonPath]):
        super().__init__()
        self.path_type = path_type

    def parse_prefix(self, parser: Parser, token: Token) -> JsonPath:
        child = parser.expression(self.precedence())
        return self.path_type(child)


class CombiningParselet(Parselet):
    def __init__(self, path_maker: Callable[[JsonPath, JsonPath], JsonPath],
                 prcendence=Precedence.binary, right_assoc=False):
        self.path_maker = path_maker
        self._precedence = prcendence
        self.right_assoc = right_assoc

    def __repr__(self):
        return f"<{type(self).__name__} {self.path_maker}, {self.precedence}>"

    def parse_infix(self, parser: Parser, left: JsonPath, token: Token
                    ) -> JsonPath:
        prec_adjust = -1 if self.right_assoc else 0
        right = parser.expression(self.precedence() + prec_adjust)
        return self.path_maker(left, right)

    def precedence(self) -> int:
        return self._precedence


class AbstractOperatorParselet(Parselet):
    def __init__(self, op: Union[str, Callable], precedence=Precedence.binary,
                 right_assoc=False):
        self.op = op
        self._precedence = precedence
        self.right_assoc = right_assoc

    def precedence(self) -> int:
        return self._precedence


class ComparisonParselet(AbstractOperatorParselet):
    def parse_infix(self, parser: Parser, left: JsonPath, token: Token
                    ) -> JsonPath:
        prec_adjust = -1 if self.right_assoc else 0
        # Hack: we want to be able to compare to a string, but normally "foo"
        # would parse to Key("foo"), so we special case it
        if parser.current().kind == Kind.string:
            right = Literal(parser.consume().payload)
        else:
            right = parser.expression(self.precedence() + prec_adjust)
        return Comparison(left, self.op, right)

    def precedence(self) -> int:
        return Precedence.comparison


class MathParselet(AbstractOperatorParselet):
    def parse_infix(self, parser: Parser, left: JsonPath, token: Token
                    ) -> JsonPath:
        prec_adjust = -1 if self.right_assoc else 0
        right = parser.expression(self.precedence() + prec_adjust)
        return Math(left, self.op, right)


class CallParselet(Parselet):
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
            maker_fn = transform_functions[fn_name]
        except KeyError:
            raise ParserError(f"No function named {fn_name}")
        return maker_fn(*args)

    def precedence(self) -> int:
        return Precedence.call


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
    Kind.open_brace: WhereParselet(),
}
infixes: dict[Kind, Parselet] = {
    Kind.child: CombiningParselet(Child.make, Precedence.child),
    Kind.desc: CombiningParselet(Descendants, Precedence.binary),
    Kind.merge: CombiningParselet(Merge, Precedence.merge),
    Kind.or_: CombiningParselet(Or, Precedence.logic),
    Kind.and_: CombiningParselet(And, Precedence.logic),
    Kind.intersect: CombiningParselet(Intersect, Precedence.intersect),
    Kind.bang: CombiningParselet(Discard, Precedence.intersect),
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
    Kind.open_brace: WhereParselet(),
}


class Parser:
    def __init__(self, text: str):
        self.text = text
        self.tokens = lex(text)
        # self.depth = 0

    def take(self, kind: Kind) -> Optional[Token]:
        if not self.tokens:
            return None
        if self.tokens[0].kind == kind:
            return self.consume()

    def _eof_token(self) -> Token:
        return Token(Kind.eof, "<EOF>", len(self.text))

    def consume(self, kind: Kind = None) -> Token:
        if self.tokens:
            token = self.tokens.pop(0)
        else:
            token = self._eof_token()

        if kind and token.kind != kind:
            raise ParserError(f"Expected {kind}, found {token.payload} "
                              f"at {token.pos}")
        return token

    def current(self) -> Token:
        return self.lookahead(0)

    def lookahead(self, distance: int) -> Optional[Token]:
        if distance >= len(self.tokens):
            return self._eof_token()
        return self.tokens[distance]

    def current_infix(self) -> Optional[Parselet]:
        return infixes.get(self.current().kind)

    def infix_precedence(self) -> int:
        infix = infixes.get(self.current().kind)
        if infix:
            return infix.precedence()
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

    def parse(self) -> JsonPath:
        path = self.expression()
        if self.tokens:
            tk = self.tokens[0]
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
    return Parser(text).parse()

