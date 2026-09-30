from midi_llm.config import load_config


def test_load_base_config():
    cfg = load_config()
    assert cfg["model"]["backbone"].startswith("meta-llama/")
    assert cfg["training"]["max_steps"]["s3_edit_lora"] == 3000


def test_load_repr_config():
    cfg = load_config("remi")
    assert cfg["representation"] == "remi"
    assert cfg["vocab_extension"] is False
    assert cfg["tokenizer_mode"] == "bpe_text"


def test_abc_config_disabled():
    cfg = load_config("abc")
    assert cfg["enabled"] is False
