"""Tests for normalize_zulip_markdown — blank line injection before tables."""
from zulipmcp.core import normalize_zulip_markdown


def test_injects_blank_before_table():
    text = "Some text\n| h1 | h2 |\n| --- | --- |\n| a | b |"
    assert normalize_zulip_markdown(text) == "Some text\n\n| h1 | h2 |\n| --- | --- |\n| a | b |"


def test_no_double_inject():
    text = "Some text\n\n| h1 | h2 |\n| --- | --- |"
    assert normalize_zulip_markdown(text) == text


def test_table_at_message_start():
    text = "| h1 | h2 |\n| --- | --- |\n| a | b |"
    assert normalize_zulip_markdown(text) == text


def test_skips_fenced_code_block():
    text = "```\nSome text\n| h1 | h2 |\n| --- | --- |\n```"
    assert normalize_zulip_markdown(text) == text


def test_skips_fenced_code_with_lang():
    text = "```python\ndata\n| h1 | h2 |\n| --- | --- |\n```"
    assert normalize_zulip_markdown(text) == text


def test_skips_tilde_fence():
    text = "~~~\n| h1 | h2 |\n| --- | --- |\n~~~"
    assert normalize_zulip_markdown(text) == text


def test_fence_close_needs_matching_char():
    text = "```\n| h1 | h2 |\n| --- | --- |\n~~~\n| h1 | h2 |\n| --- | --- |\n```"
    assert normalize_zulip_markdown(text) == text


def test_fence_close_needs_matching_length():
    text = "````\n| h1 | h2 |\n| --- | --- |\n```\nstill inside\n````"
    assert normalize_zulip_markdown(text) == text


def test_fence_close_rejects_content_after():
    text = "```\n| h1 | h2 |\n| --- | --- |\n```python\nstill fenced\n```"
    assert normalize_zulip_markdown(text) == text


def test_table_after_fenced_block():
    text = "```\ncode\n```\nSome text\n| h1 | h2 |\n| --- | --- |"
    assert normalize_zulip_markdown(text) == "```\ncode\n```\nSome text\n\n| h1 | h2 |\n| --- | --- |"


def test_skips_indented_code():
    text = "paragraph\n    | h1 | h2 |\n    | --- | --- |"
    assert normalize_zulip_markdown(text) == text


def test_skips_tab_indented_code():
    text = "paragraph\n\t| h1 | h2 |\n\t| --- | --- |"
    assert normalize_zulip_markdown(text) == text


def test_skips_blockquote():
    text = "paragraph\n> | h1 | h2 |\n> | --- | --- |"
    assert normalize_zulip_markdown(text) == text


def test_consecutive_tables_separated():
    text = "| h1 | h2 |\n| --- | --- |\n| a | b |\n| h3 | h4 |\n| --- | --- |\n| c | d |"
    assert normalize_zulip_markdown(text) == (
        "| h1 | h2 |\n| --- | --- |\n| a | b |\n\n| h3 | h4 |\n| --- | --- |\n| c | d |"
    )


def test_pipe_in_text_no_separator():
    text = "Choose this | that\nNext paragraph"
    assert normalize_zulip_markdown(text) == text


def test_no_next_line():
    text = "Some text\n| h1 | h2 |"
    assert normalize_zulip_markdown(text) == text


def test_alignment_separators():
    text = "Text\n| left | center | right |\n| :--- | :---: | ---: |\n| a | b | c |"
    assert normalize_zulip_markdown(text) == (
        "Text\n\n| left | center | right |\n| :--- | :---: | ---: |\n| a | b | c |"
    )


def test_no_border_pipes():
    text = "Text\nh1 | h2\n--- | ---\na | b"
    assert normalize_zulip_markdown(text) == "Text\n\nh1 | h2\n--- | ---\na | b"


def test_after_heading():
    text = "## Heading\n| h1 | h2 |\n| --- | --- |"
    expected = "## Heading\n\n| h1 | h2 |\n| --- | --- |"
    assert normalize_zulip_markdown(text) == expected


def test_after_hr():
    text = "text\n\n---\n| h1 | h2 |\n| --- | --- |"
    expected = "text\n\n---\n\n| h1 | h2 |\n| --- | --- |"
    assert normalize_zulip_markdown(text) == expected


def test_spoiler_block_skipped():
    text = "```spoiler Details\n| h1 | h2 |\n| --- | --- |\n```"
    assert normalize_zulip_markdown(text) == text


def test_empty_content():
    assert normalize_zulip_markdown("") == ""


def test_no_tables():
    text = "Just some normal text\nwith multiple lines\nand no tables"
    assert normalize_zulip_markdown(text) == text


def test_multiple_tables_in_message():
    text = (
        "First section\n| a | b |\n| --- | --- |\n| 1 | 2 |\n\n"
        "Second section\n| c | d |\n| --- | --- |\n| 3 | 4 |"
    )
    expected = (
        "First section\n\n| a | b |\n| --- | --- |\n| 1 | 2 |\n\n"
        "Second section\n\n| c | d |\n| --- | --- |\n| 3 | 4 |"
    )
    assert normalize_zulip_markdown(text) == expected


def test_separator_with_minimal_dashes():
    text = "Text\n| h1 | h2 |\n| - | - |\n| a | b |"
    assert normalize_zulip_markdown(text) == "Text\n\n| h1 | h2 |\n| - | - |\n| a | b |"


def test_real_world_llm_output():
    text = (
        "Here are the results:\n"
        "| Name | Status | Score |\n"
        "| --- | --- | --- |\n"
        "| Alice | Active | 95.2 |\n"
        "| Bob | Pending | 87.0 |"
    )
    expected = (
        "Here are the results:\n\n"
        "| Name | Status | Score |\n"
        "| --- | --- | --- |\n"
        "| Alice | Active | 95.2 |\n"
        "| Bob | Pending | 87.0 |"
    )
    assert normalize_zulip_markdown(text) == expected
