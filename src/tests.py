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


def test_parse_every():
    assert parse("foo.*") == Child(Key("foo"), Every())


def test_parse_descendants():
    assert parse("foo..bar") == Descendants(Key("foo"), Key("bar"))


def test_parse_choice():
    assert parse("foo||bar") == Choice(Key("foo"), Key("bar"))


def test_parse_choice_and_merge():
    p = parse("foo||bar|baz")
    assert p == Choice(Key("foo"), Merge(Key("bar"), Key("baz")))

    p = parse("foo|bar||baz")
    assert p == Choice(Merge(Key("foo"), Key("bar")), Key("baz"))


def test_intersect():
    assert parse("foo & bar") == Intersect(Key("foo"), Key("bar"))


def test_parse_index():
    assert parse("[522]") == Index(522)
    assert parse("[-99]") == Index(-99)


def test_parse_slice():
    assert parse("[4:9]") == Index(slice(4, 9))
    assert parse("[4:9:3]") == Index(slice(4, 9, 3))
    assert parse("[-4:9]") == Index(slice(-4, 9))
    assert parse("[4:-100]") == Index(slice(4, -100))
    assert parse("[4:9:-3]") == Index(slice(4, 9, -3))
    assert parse("[4:]") == Index(slice(4, None))
    assert parse("[4::-3]") == Index(slice(4, None, -3))


def test_parse_child_index():
    assert parse("foo[5]") == Child(Key("foo"), Index(5))
    assert parse("foo.[5]") == Child(Key("foo"), Index(5))


def test_parse_parens():
    p = parse("a.b|c.d")
    assert p == Merge(Child(Key("a"), Key("b")), Child(Key("c"), Key("d")))

    p = parse("(a.b)|(c.d)")
    assert p == Merge(Child(Key("a"), Key("b")), Child(Key("c"), Key("d")))

    p = parse("a.(b|c).d")
    assert p == Child(Key('a'), Child(Merge(Key('b'), Key('c')), Key('d')))


def test_parse_paren_mismatch():
    with pytest.raises(ParserError):
        parse("(a.b")

    with pytest.raises(ParserError):
        parse("a.b)")


def test_parse_fn_maker():
    assert parse("foo.len()") == Child(Key("foo"), Func(len))


def test_parse_path_maker():
    assert parse("foo.parent()") == Child(Key("foo"), Parent())


def test_parse_compare():
    assert parse("name == 'foo'") == Child(Key("name"), Comparison("==", "foo"))
    assert parse("foo == 2") == Child(Key("foo"), Comparison("==", 2))
    assert parse("foo ==  -3.4") == Child(Key("foo"), Comparison("==", -3.4))
    assert parse("<= 5") == Comparison("<=", 5)
    assert parse("<= 5 .color") == Child(Comparison("<=", 5), Key("color"))

    p = parse("* <- (type == 'car')")
    assert p == Where(Every(), Child(Key("type"), Comparison("==", "car")))

    p = parse("* <- (type == 'car').color")
    assert p == Child(Where(Every(), Child(Key("type"), Comparison("==", "car"))), Key("color"))

    p = parse("* <- look .color == 10")
    assert p == Child(
        Where(
            Every(),
            Key("look")
        ),
        Child(
            Key("color"),
            Comparison("==", 10)
        )
    )


def test_parse_compare_errors():
    with pytest.raises(ParserError):
        parse("name ==")

    with pytest.raises(ParserError):
        parse("name == .")


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


def test_find_weird_compare():
    assert parse("!= 5").values({}) == [{}]
    assert parse("<= 5 .color").values({}) == []
