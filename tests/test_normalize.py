import pytest

from entity_cascade.data.normalize import (
    build_brand_vocab,
    canonical_code,
    extract_brand,
    extract_codes,
    normalize_brand,
    normalize_text,
    tokenize,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Sony® Turntable – PSLX350H", "sony turntable - pslx350h"),
        ("Canon™ EOS", "canon eos"),
        ("  Multiple   spaces\tand\nnewlines ", "multiple spaces and newlines"),
        ('10" LCD TV', "10in lcd tv"),
        ("10-inch monitor", "10in monitor"),
        ("19.5 inches wide", "19.5in wide"),
        ("2 GB card", "2gb card"),
        ("2-GB card", "2gb card"),
        ("512MB", "512mb"),
        ("8.1 Megapixels", "8.1mp"),
        ("100-240V adapter", "100-240v adapter"),
        ("12 Watts", "12w"),
        ("3 in 1 printer", "3 in 1 printer"),
        ("ＡＢＣ １２３", "abc 123"),  # full-width forms (NFKC)
    ],
)
def test_normalize_text(raw: str, expected: str) -> None:
    assert normalize_text(raw) == expected


def test_normalize_is_idempotent() -> None:
    s = 'Sony® 2 GB – 10" DMC-FZ18K'
    once = normalize_text(s)
    assert normalize_text(once) == once


def test_tokenize_emits_split_and_joined_codes() -> None:
    assert tokenize("DMC-FZ18K camera") == ["dmc", "fz18k", "dmcfz18k", "camera"]


def test_tokenize_does_not_join_plain_words() -> None:
    assert tokenize("black-and-white") == ["black", "and", "white"]


def test_canonical_code() -> None:
    assert canonical_code("DMC-FZ18/K") == "dmcfz18k"


@pytest.mark.parametrize(
    ("text", "codes"),
    [
        ("Linksys EtherFast EZXS88W Ethernet Switch - EZXS88W", ["ezxs88w"]),
        ("Panasonic DMC-FZ18K Lumix 8.1MP", ["dmcfz18k"]),
        ("Sony Turntable - PSLX350H", ["pslx350h"]),
        ("Sony 4GB Memory Stick 720p 1080p", []),  # units are not model codes
        ("Plain words only", []),
        ("Model 12345 numeric only", []),
        ("SBH-50 and SBH50", ["sbh50"]),  # same canonical code, listed once
    ],
)
def test_extract_codes(text: str, codes: list[str]) -> None:
    assert extract_codes(text) == codes


def test_normalize_brand_strips_company_suffix() -> None:
    assert normalize_brand("Hewlett-Packard Inc.") == "hewlett packard"
    assert normalize_brand("SONY") == "sony"


def test_extract_brand() -> None:
    vocab = build_brand_vocab(["Sony", "LINKSYS", "Hewlett-Packard Inc", None, ""])
    assert extract_brand("Sony Turntable - PSLX350H", vocab) == "sony"
    assert extract_brand('Hewlett-Packard 15" laptop', vocab) == "hewlett packard"
    assert extract_brand("Black Linksys switch", vocab) == "linksys"  # not at the start
    assert extract_brand("Unknown thing", vocab) is None
