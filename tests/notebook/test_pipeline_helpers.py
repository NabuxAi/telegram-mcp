from telegram_notebook.pipeline import IngestionPipeline


def test_combine_text_parts():
    assert IngestionPipeline._combine_text_parts("transcript", "caption") == "transcript\n\ncaption"
    assert IngestionPipeline._combine_text_parts("only transcript", None) == "only transcript"
    assert IngestionPipeline._combine_text_parts(None, "only caption") == "only caption"
    assert IngestionPipeline._combine_text_parts(None, None) == ""


def test_combine_dedupes_identical_caption():
    assert IngestionPipeline._combine_text_parts("same", "same") == "same"


def test_channel_storage_name_sanitizes():
    assert IngestionPipeline._channel_storage_name("https://t.me/example") == "example"
    assert IngestionPipeline._channel_storage_name("https://t.me/a.b c") == "a_b_c"
    assert IngestionPipeline._channel_storage_name("https://t.me/") == "t_me"
