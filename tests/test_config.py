from midi_llm.config import load_config


def test_load_base_config():
    cfg = load_config()
    assert cfg["model"]["backbone"].startswith("meta-llama/")
    assert cfg["training"]["max_steps"]["s3_edit_lora"] == 3000


def test_load_repr_config():
    cfg = load_config("remi")
    assert cfg["representation"] == "remi"
    assert cfg["vocab_extension"] is True
