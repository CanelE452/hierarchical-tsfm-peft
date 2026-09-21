from peft import LoraConfig, get_peft_model
from types import MethodType


def _get_shared_embeddings(model):
    return model.shared


def attach_lora(model):
    # PEFT 0.21 inspects tied embeddings through this API. Chronos-Bolt 2.3.2
    # defines the shared embedding but inherits the unimplemented T5 accessor.
    # This exposes existing weights; neither weights nor the forward path change.
    model.get_input_embeddings = MethodType(_get_shared_embeddings, model)
    config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, target_modules=["q", "v"], bias="none")
    model = get_peft_model(model, config)
    names = {name: p.numel() for name, p in model.named_parameters() if p.requires_grad}
    if sum(names.values()) != 294912:
        raise RuntimeError(f"BLOCKED_IMPLEMENTATION: LoRA structure/count mismatch: {sum(names.values())}; {names}")
    if not all("lora_" in name for name in names):
        raise RuntimeError("Unfrozen base weight")
    return model, names
