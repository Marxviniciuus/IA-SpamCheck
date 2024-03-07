from utils.text import clean_text

def test_should_be_able_to_parse_lower_case():
    text = 'RAFIRA'

    assert clean_text(text) == 'rafira'


def test_should_be_able_to_replace_special_chars():
    text = '@RAFIRA$'

    assert clean_text(text) == ' rafira '


def test_should_be_able_to_replace_punctuations():
    text = 'meu nome eh rafira! e o seu?'

    assert clean_text(text) == 'meu nome eh rafira e o seu '


def test_should_be_able_to_replace_large_blank_spaces():
    text = 'rafira    nao gosta     de        java'

    assert clean_text(text) == 'rafira nao gosta de java'
