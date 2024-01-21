import pytest
from jsonpathfx import *


def test_parse_ascii_key():
    assert parse("foo") == Key("foo")


def test_parse_child():
    assert parse("foo.bar") == Child(Key("foo"), Key("bar"))


def test_parse_child_ws():
    assert parse("foo . bar") == Child(Key("foo"), Key("bar"))


def test_parse_unicode_key():
    assert parse("ἰού.πάντʼ") == Child(Key("ἰού"), Key("πάντʼ"))


def test_parse_no_op():
    with pytest.raises(ParserError):
        parse("foo bar")


def test_parse_quoted_key():
    assert parse("'foo.bar'") == Key("foo.bar")
    assert parse('"foo.bar"') == Key("foo.bar")
    assert parse("'foo\\'bar'") == Key("foo'bar")
    assert parse('"foo\\"bar"') == Key('foo"bar')

    with pytest.raises(ParserError):
        parse("'foo")

    with pytest.raises(ParserError):
        parse('"foo')

    with pytest.raises(ParserError):
        parse("foo'")

    with pytest.raises(ParserError):
        parse('foo"')


def test_parse_bracketed_unquoted_key():
    assert parse("[foo]") == Key("foo")


def test_bracketed_subexpression():
    assert parse("foo[bar.baz]") == Child(Key("foo"),
                                          Child(Key("bar"), Key("baz")))


def test_parse_bracketed_quoted_key():
    assert parse("['foo']") == Key("foo")
    assert parse('["foo"]') == Key("foo")
    assert parse('["[foo]"]') == Key("[foo]")

    with pytest.raises(ParserError):
        parse('["foo]"')


def test_parse_bracketed_child_key():
    assert parse("foo[bar]") == Child(Key("foo"), Key("bar"))
    assert parse("foo[bar]") == Child(Key("foo"), Key("bar"))
    assert parse("foo.[bar]") == Child(Key("foo"), Key("bar"))
    assert parse("[foo][bar]") == Child(Key("foo"), Key("bar"))


def test_parse_root():
    assert parse("$") == Root()
    assert parse("$.foo") == Key("foo")
    assert parse("foo.$") == Root()


def test_parse_this():
    assert parse("@") == This()
    assert parse("foo.@") == Key("foo")


def test_parse_where():
    assert parse("foo <- bar") == Where(Key("foo"), Key("bar"))


def test_parse_where_multi():
    p = parse("foo <- bar <- baz")
    assert p == Where(Where(Key("foo"), Key("bar")), Key("baz"))


def test_parse_every():
    assert parse("foo.*") == Child(Key("foo"), Every())


def test_parse_descendants():
    assert parse("foo..bar") == Descendants(Key("foo"), Key("bar"))


def test_parse_choice():
    assert parse("foo||bar") == Or(Key("foo"), Key("bar"))


def test_parse_choice_and_merge():
    p = parse("foo|bar|baz")
    assert p == Merge(Merge(Key("foo"), Key("bar")), Key("baz"))

    p = parse("foo||bar||baz")
    assert p == Or(Or(Key("foo"), Key("bar")), Key("baz"))

    p = parse("foo||bar|baz")
    assert p == Merge(Or(Key("foo"), Key("bar")), Key("baz"))

    p = parse("foo|bar||baz")
    assert p == Merge(Key("foo"), Or(Key("bar"), Key("baz")))


def test_intersect():
    assert parse("foo & bar") == Intersect(Key("foo"), Key("bar"))


def test_parse_index():
    assert parse("[522]") == Index(522)
    assert parse("[-99]") == Index(-99)

    with pytest.raises(ParserError):
        parse("[1.2]")


def test_parse_slice():
    assert parse("[4:9]") == Index(slice(4, 9))
    assert parse("[4:9:3]") == Index(slice(4, 9, 3))
    assert parse("[-4:9]") == Index(slice(-4, 9))
    assert parse("[4:-100]") == Index(slice(4, -100))
    assert parse("[4:9:-3]") == Index(slice(4, 9, -3))
    assert parse("[4:]") == Index(slice(4, None))
    assert parse("[4::-3]") == Index(slice(4, None, -3))
    assert parse("[::-3]") == Index(slice(None, None, -3))
    assert parse("[::]") == Index(slice(None, None, None))


def test_parse_child_index():
    assert parse("foo[5]") == Child(Key("foo"), Index(5))
    assert parse("foo.[5]") == Child(Key("foo"), Index(5))


def test_parse_parens():
    p = parse("a.b|c.d")
    assert p == Merge(Child(Key("a"), Key("b")), Child(Key("c"), Key("d")))

    p = parse("(a.b)|(c.d)")
    assert p == Merge(Child(Key("a"), Key("b")), Child(Key("c"), Key("d")))

    p = parse("a.(b|c).d")
    assert p == Child(Child(Key('a'), Merge(Key('b'), Key('c'))), Key('d'))


def test_parse_paren_mismatch():
    with pytest.raises(ParserError):
        parse("(a.b")

    with pytest.raises(ParserError):
        parse("a.b)")


def test_parse_len():
    assert parse("foo.len()") == Child(Key("foo"), FilterFunction(len))


def test_parse_path_maker():
    assert parse("foo.parent()") == Child(Key("foo"), Parent())


def test_parse_compare():
    assert parse("name == 'foo'") == Comparison(Key("name"), "==", Literal("foo"))
    assert parse("foo == 2") == Comparison(Key("foo"), "==", Literal(2))
    assert parse("foo ==  -3.4") == Comparison(Key("foo"), "==", Literal(-3.4))

    p = parse("* <- (type == 'car')")
    assert p == Where(Every(), Comparison(Key("type"), "==", Literal("car")))

    p = parse("* <- (type == 'car').color")
    assert p == Where(
        Every(),
        Child(
            Comparison(Key("type"), "==", Literal("car")),
            Key("color")
        )
    )

    p = parse("* <- look .color == 10")
    assert p == Where(
        Every(),
        Comparison(
            Child(
                Key("look"),
                Key("color")
            ),
            "==",
            Literal(10)
        )
    )


def test_parse_compare_compound_path():
    p = parse("$.*.size > 15")
    assert p == Comparison(
        Child(
            Every(),
            Key("size")
        ),
        ">",
        Literal(15)
    )


def test_parse_compare_errors():
    with pytest.raises(ParserError):
        parse("name ==")

    with pytest.raises(ParserError):
        parse("name == .")


def test_parse_bind():
    p = parse("foo.x:*.bar")
    assert p == Child(
        Child(
            Key("foo"),
            Bind(Every(), "x"),
        ),
        Key("bar")
    )


def test_parse_math():
    assert parse("foo + 5") == Math(Key("foo"), "+", Literal(5))
    assert parse("foo - bar") == Math(Key("foo"), "-", Key("bar"))
    assert parse("6.5 * bar") == Math(Literal(6.5), "-", Key("bar"))


def test_parse_math_every():
    assert parse("* * *") == Math(Every(), "*", Every())


def test_find_key():
    domain = {
        "foo": 10,
        "bar": 20,
        "baz": [1, 2, 3],
        "husk:stats": 30
    }
    assert parse("foo").values(domain) == [10]
    assert parse("[bar]").values(domain) == [20]
    assert parse("baz").values(domain) == [[1, 2, 3]]
    assert parse("'husk:stats'").values(domain) == [30]
    assert parse("bazz").values(domain) == []


def test_find_index():
    domain = {
        "foo": [5, 7, 10],
    }
    assert parse("foo[1]").values(domain) == [7]
    assert parse("foo[-1]").values(domain) == [10]
    assert parse("foo[3]").values(domain) == []
    assert parse("bar[3]").values(domain) == []


def test_find_slice():
    domain = {
        "foo": [5, 10, 15, 20, 25, 30],
    }
    assert parse("foo[1:2]").values(domain) == [10]
    assert parse("foo[3:]").values(domain) == [20, 25, 30]
    assert parse("foo[::2]").values(domain) == [5, 15, 25]


def test_find_fn():
    domain = {
        "foo": ["c", "t", "e", "p", "b"],
    }
    assert parse("foo.len()").values(domain) == [5]
    assert parse("$.keys()").values(domain) == [["foo"]]
    assert parse("foo.sorted()").values(domain) == [["b", "c", "e", "p", "t"]]


def test_find_parent():
    domain = {
        "foo": {
            "bar": 20,
            "baz": 30
        }
    }
    target = domain["foo"]
    assert parse("foo.bar").values(domain) == [20]
    assert parse("foo.bar.parent()").values(domain) == [target]
    assert parse("foo.bar.parent().parent()").values(domain) == [domain]


def test_root_parent():
    domain = {
        "foo": {
            "bar": 20,
            "baz": 30
        }
    }
    assert parse("$.parent()").values(domain) == []


def test_find_where_sibling():
    domain = [
        {"name": "foo", "size": 3},
        {"name": "bar"},
        {"name": "baz", "size": 6},
    ]
    assert parse("(* <- size).name").values(domain) == ["foo", "baz"]


def test_find_every():
    domain = {
        "books": {
            "foo": {"a": 10, "b": 20},
            "bar": {"a": 30, "b": 40},
            "baz": {"a": 50, "b": 60},
        }
    }
    assert parse("$.books[*].b").values(domain) == [20, 40, 60]


def test_find_descendents():
    domain = {
        "alfa": {
            "bravo": "a",
            "charlie": "b",
            "delta": "c"
        },
        "echo": {
            "foxtrot": "d",
            "golf": "e",
            "mike": "j"
        },
        "hotel": {
            "india": "f",
            "juliet": "g",
            "kilo": {
                "lima": "h",
                "mike": "i"
            }
        }
    }
    assert parse("$..mike").values(domain) == ["j", "i"]


def test_find_merge():
    domain = {
        "geo": {
            "detail": [
                {"name": "a"},
                {"name": "b"},
                {"name": "c"}
            ],
            "point": [
                {"name": "d"},
            ],
            "vertex": [
                {"name": "e"},
                {"name": "f"},
            ]
        }
    }
    p = parse("geo.(detail|point|vertex).*.name")
    assert p.values(domain) == ["a", "b", "c", "d", "e", "f"]


def test_find_compare():
    domain = {
        "foo": {"size": 10},
        "bar": {"size": 20},
        "baz": {"size": 30},
    }
    assert parse("$.*.size > 15").values(domain) == [20, 30]


def test_find_compare2():
    domain = {
        "foo": {"type": "car", "color": "red"},
        "bar": {"type": "boat", "color": "blue"},
        "baz": {"type": "car", "color": "green"},
    }
    p = parse("(* <- (type == 'car')).color")
    assert p.values(domain) == ["red", "green"]


def test_compare_to_path():
    domain = {
        "foo": [1, 2, 3, 4, 5, 6, 7],
        "cutoff": 4
    }
    p = parse("foo.* < cutoff")
    assert p.values(domain) == [1, 2, 3]


def test_compare_type_mismatch():
    domain = {
        "foo": [1, 2, 3, 4, 5, 6, 7],
        "cutoff": "x"
    }
    p = parse("foo.* < cutoff")
    assert p.values(domain) == []


def test_match_path():
    domain = {
        "alfa": {
            "bravo": "a",
            "charlie": "b",
            "delta": "c"
        },
        "echo": {
            "foxtrot": "d",
            "golf": "e",
        },
        "hotel": {
            "india": "f",
            "juliet": "g",
            "kilo": {
                "lima": "h",
                "mike": "i"
            }
        }
    }
    p = parse("*.*")
    assert p.values(domain) == ['a', 'b', 'c', 'd', 'e', 'f', 'g',
                                 {'lima': 'h', 'mike': 'i'}]

    assert [m.path() for m in p.find(domain)] == [
        ("alfa", "bravo"), ("alfa", "charlie"), ("alfa", "delta"),
        ("echo", "foxtrot"), ("echo", "golf"), ("hotel", "india"),
        ("hotel", "juliet"), ("hotel", "kilo")
    ]


def test_match_bindings():
    domain = {
        "alfa": {
            "bravo": "a",
            "charlie": "b",
            "delta": "c"
        },
        "echo": {
            "foxtrot": "d",
            "golf": "e",
            "mike": "j"
        },
        "hotel": {
            "india": "f",
            "juliet": "g",
            "kilo": {
                "lima": "h",
                "mike": "i"
            }
        }
    }
    p = parse("top:*..mike")
    assert p.values(domain) == ["j", "i"]
    assert [m.bindings() for m in p.find(domain)] == [
        {"top": "echo"},
        {"top": "hotel"}
    ]


def test_match_index_bindings():
    domain = {
        "alfa": ["a", "b", "c", "d"],
        "echo": ["c", "d", "e", "f"],
        "hotel": ["d", "e", "f", "g"]
    }
    p = parse("k:*.(x:* == 'd')")
    assert [m.bindings() for m in p.find(domain)] == [
        {"k": "alfa", "x": 3},
        {"k": "echo", "x": 1},
        {"k": "hotel", "x": 0}
    ]


def test_merge_binding():
    domain = {
        "geo": {
            "detail": [
                {"name": "a"},
                {"name": "b"},
                {"name": "c"}
            ],
            "point": [
                {"name": "d"},
            ],
            "vertex": [
                {"name": "e"},
                {"name": "f"},
            ]
        }
    }
    p = parse("geo.comp:(detail|point|vertex).*.name")
    comps = [m.bindings()["comp"] for m in p.find(domain)]
    assert comps == ["detail", "detail", "detail", "point", "vertex", "vertex"]


def test_find_math():
    domain = {
        "foo": [1, 2, 3, 4, 5],
        "bar": 10
    }
    p = parse("foo.* * bar")
    assert p.values(domain) == [10, 20, 30, 40, 50]
