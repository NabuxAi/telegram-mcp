from telegram_notebook import config


def test_upsert_env_values_updates_and_appends(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("A=1\n# a comment\n\nB=2\n", encoding="utf-8")
    monkeypatch.setattr(config, "ENV_PATH", env_file)

    config.upsert_env_values({"B": "22", "C": "3"})

    lines = env_file.read_text(encoding="utf-8").splitlines()
    assert "A=1" in lines          # untouched
    assert "# a comment" in lines  # comment preserved
    assert "" in lines             # blank line preserved
    assert "B=22" in lines         # existing key updated
    assert "C=3" in lines          # new key appended


def test_upsert_env_values_blanks_none(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("TOKEN=secret\n", encoding="utf-8")
    monkeypatch.setattr(config, "ENV_PATH", env_file)

    config.upsert_env_values({"TOKEN": None})
    assert "TOKEN=" in env_file.read_text(encoding="utf-8").splitlines()
