import pytest
from jsonpathfx import (parse, BracketedKey, Child, Choice, Descendants, Every,
                        Func, Index, Intersect, Key, Merge, Parent, ParserError,
                        Root, This, Where)


def test_parse_ascii_key():
    p = parse("foo")
    assert p == Key("foo")


def test_parse_child():
    p = parse("foo.bar")
    assert p == Child(Key("foo"), Key("bar"))


def test_parse_child_ws():
    p = parse("foo . bar")
    assert p == Child(Key("foo"), Key("bar"))


def test_parse_unicode_key():
    p = parse("ἰού.πάντʼ")
    assert p == Child(Key("ἰού"), Key("πάντʼ"))


def test_parse_quoted_key():
    p = parse("'foo.bar'")
    assert p == Key("foo.bar")

    p = parse('"foo.bar"')
    assert p == Key("foo.bar")

    with pytest.raises(ParserError):
        parse("'foo")

    with pytest.raises(ParserError):
        parse('"foo')

    with pytest.raises(ParserError):
        parse("foo'")

    with pytest.raises(ParserError):
        parse('foo"')


def test_parse_bracketed_unquoted_key():
    p = parse("[foo]")
    assert p == BracketedKey("foo")

    with pytest.raises(ParserError):
        parse("[foo.bar]")


def test_parse_bracketed_quoted_key():
    p = parse("['foo']")
    assert p == BracketedKey("foo")

    p = parse('["foo"]')
    assert p == BracketedKey("foo")

    p = parse('["[foo]"]')
    assert p == BracketedKey("[foo]")

    with pytest.raises(ParserError):
        parse('["foo]"')


def test_parse_bracketed_child_key():
    p = parse("foo[bar]")
    assert p == Child(Key("foo"), BracketedKey("bar"))

    p = parse("foo[bar]")
    assert p == Child(Key("foo"), BracketedKey("bar"))

    p = parse("foo.[bar]")
    assert p == Child(Key("foo"), BracketedKey("bar"))

    p = parse("[foo][bar]")
    assert p == Child(BracketedKey("foo"), BracketedKey("bar"))


def test_parse_root():
    p = parse("$")
    assert p == Root()

    p = parse("$.foo")
    assert p == Key("foo")


def test_parse_root_as_child():
    p = parse("foo.$")
    assert p == Root()


def test_parse_this():
    p = parse("@")
    assert p == This()


def test_parse_this_as_child():
    p = parse("foo.@")
    assert p == Key("foo")


def test_parse_where():
    p = parse("foo <- bar")
    assert p == Where(Key("foo"), Key("bar"))


def test_parse_every():
    p = parse("foo.*")
    assert p == Child(Key("foo"), Every())


def test_parse_descendants():
    p = parse("foo..bar")
    assert p == Descendants(Key("foo"), Key("bar"))


def test_parse_choice():
    p = parse("foo||bar")
    assert p == Choice(Key("foo"), Key("bar"))


def test_parse_choice_and_merge():
    p = parse("foo||bar|baz")
    assert p == Choice(Key("foo"), Merge(Key("bar"), Key("baz")))

    p = parse("foo|bar||baz")
    assert p == Choice(Merge(Key("foo"), Key("bar")), Key("baz"))


def test_intersect():
    p = parse("foo & bar")
    assert p == Intersect(Key("foo"), Key("bar"))


def test_parse_index():
    p = parse("[522]")
    assert p == Index(522)

    p = parse("[-99]")
    assert p == Index(-99)


def test_parse_slice():
    p = parse("[4:9]")
    assert p == Index(slice(4, 9))

    p = parse("[4:9:3]")
    assert p == Index(slice(4, 9, 3))

    p = parse("[-4:9]")
    assert p == Index(slice(-4, 9))

    p = parse("[4:-100]")
    assert p == Index(slice(4, -100))

    p = parse("[4:9:-3]")
    assert p == Index(slice(4, 9, -3))

    p = parse("[4:]")
    assert p == Index(slice(4, None))

    p = parse("[4::-3]")
    assert p == Index(slice(4, None, -3))


def test_parse_child_index():
    p = parse("foo[5]")
    assert p == Child(Key("foo"), Index(5))

    p = parse("foo.[5]")
    assert p == Child(Key("foo"), Index(5))


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

