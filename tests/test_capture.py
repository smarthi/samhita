"""Network-free capture-tool test: a tiny, randomly-initialized Qwen2
model exercises exactly the same monkeypatch/cache-read path
`capture_activations` uses against the real `Qwen/Qwen2.5-0.5B-Instruct`
(verified manually end-to-end, see docs/design.md status note), without
needing a download or a tokenizer in CI.
"""

from __future__ import annotations

import torch

from samhita.capture import capture_activations


def _tiny_qwen2_model():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    return Qwen2ForCausalLM(config)


def test_capture_activations_end_to_end_on_tiny_model():
    torch.manual_seed(0)
    model = _tiny_qwen2_model()
    input_ids = torch.randint(0, model.config.vocab_size, (1, 10))

    result = capture_activations(model=model, input_ids=input_ids)

    assert len(result.layers) == model.config.num_hidden_layers
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    for layer in result.layers:
        assert layer.q.shape == (model.config.num_attention_heads, 10, head_dim)
        assert layer.k.shape == (model.config.num_key_value_heads, 10, head_dim)
        assert layer.v.shape == (model.config.num_key_value_heads, 10, head_dim)

    assert result.manifest["capture_path"] == "post_rope"
    assert result.manifest["seq_len"] == 10
    assert result.manifest["num_layers"] == 2
