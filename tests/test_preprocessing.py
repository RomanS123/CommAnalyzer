import pytest

from app.utils.preprocessing import (

    rm_link, rm_punct2, rm_html, space_bt_punct, rm_number,

    rm_whitespaces, rm_nonascii, rm_emoji, spell_correction,

    clean_pipeline, preprocess_pipeline

)



def test_rm_link():

    assert rm_link("Visit https://example.com") == "Visit "



def test_rm_punct2():

    assert rm_punct2("Hello, world!") == "Hello  world "



def test_rm_html():

    assert rm_html("<p>Text</p>") == "Text"



def test_space_bt_punct():
    print(space_bt_punct("Hello,world!"))
    assert space_bt_punct("Hello,world!") == "Hello , world ! "



def test_rm_number():

    assert rm_number("I have 2 apples") == "I have  apples"



def test_clean_pipeline():

    text = "Visit https://example.com! <p>Hello123</p>😊"

    expected = "Visit Hello"

    assert clean_pipeline(text) == expected



def test_preprocess_pipeline():

    text = "I have 2 apples and some bananas"

    result = preprocess_pipeline(text)

    assert isinstance(result, str)

    assert "apple" in result

