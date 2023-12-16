import pytest
from jsonpathfx import (parse, Child, Choice, Descendants, Every, Func, Index,
                        Intersect, Key, Merge, Parent, ParserError, Root, This,
                        Where)


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
    p = parse("foo.len()")
    assert p == Child(Key("foo"), Func(len))


def test_parse_path_maker():
    p = parse("foo.parent()")
    assert p == Child(Key("foo"), Parent())


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
        {"name": "foo", "size": 2, "extra": True},
        {"name": "bar", "size": 2},
        {"name": "baz", "size": 2, "extra": True},
    ]
    assert parse("(* <- extra).name").values(domain) == ["foo", "baz"]
