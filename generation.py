"""Model-agnostic autoregressive generation helpers for ToriiGate nodes.

The custom nodes intentionally do not load, cache, move, or identify models.
ComfyUI's loader owns the model object and its ``ModelPatcher`` lifecycle.  This
module only consumes a model-like object and performs the token loop needed by
the two ToriiGate nodes.

The small adapter layer is deliberately duck-typed.  Different ComfyUI
loaders expose the underlying causal model as ``model``, ``cond_stage_model``
or directly, while tokenizer access is commonly exposed as ``tokenizer``.
No backend-specific import is required here.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Callable, Mapping, MutableMapping, Optional

logger = logging.getLogger("ToriiGate")


def _torch():
    """Import torch lazily so the node package can be inspected without it."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised in non-ComfyUI installs
        raise RuntimeError(
            "ToriiGate generation requires the PyTorch runtime supplied by ComfyUI."
        ) from exc
    return torch


def _is_tensor(value: Any) -> bool:
    try:
        return bool(value is not None and value.__class__.__module__.split(".")[0] == "torch")
    except Exception:
        return False


def _unwrap_candidates(model: Any):
    """Yield likely callable model wrappers without assuming a backend type."""
    seen = set()
    pending = [model]
    while pending:
        candidate = pending.pop(0)
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        yield candidate
        for name in ("model", "cond_stage_model", "inner_model", "transformer", "language_model"):
            try:
                child = getattr(candidate, name, None)
            except Exception:
                child = None
            if child is not None and child is not candidate:
                pending.append(child)


def resolve_generation_model(model: Any) -> Any:
    """Return the object implementing a causal forward/generate operation."""
    for candidate in _unwrap_candidates(model):
        if callable(candidate) or callable(getattr(candidate, "generate", None)) or callable(getattr(candidate, "generate_tokens", None)):
            return candidate
    raise TypeError(
        "The connected CLIP/model does not expose a causal generation interface. "
        "Connect a ToriiGate-compatible ComfyUI model from its native loader."
    )


def resolve_tokenizer(model: Any) -> Any:
    """Find a tokenizer on a standard ComfyUI model wrapper."""
    for candidate in _unwrap_candidates(model):
        for name in ("tokenizer", "processor", "text_processor"):
            try:
                tokenizer = getattr(candidate, name, None)
            except Exception:
                tokenizer = None
            if tokenizer is not None and (
                callable(tokenizer)
                or callable(getattr(tokenizer, "__call__", None))
                or callable(getattr(tokenizer, "decode", None))
                or callable(getattr(tokenizer, "batch_decode", None))
                or callable(getattr(tokenizer, "apply_chat_template", None))
                or callable(getattr(tokenizer, "encode", None))
            ):
                return tokenizer
    return None


def _model_device(model: Any):
    """Read a model's existing device; never moves the model."""
    for candidate in _unwrap_candidates(model):
        try:
            parameters = candidate.parameters()
            return next(parameters).device
        except (AttributeError, StopIteration, TypeError, RuntimeError):
            continue
    return None


def _move_inputs_to_model_device(value: Any, device: Any) -> Any:
    if device is None:
        return value
    if _is_tensor(value):
        try:
            return value.to(device)
        except Exception:
            return value
    if isinstance(value, Mapping):
        return {key: _move_inputs_to_model_device(item, device) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        converted = [_move_inputs_to_model_device(item, device) for item in value]
        return type(value)(converted)
    return value


def _get_output_value(output: Any, *names: str) -> Any:
    if isinstance(output, (tuple, list)):
        # Common lightweight model adapters return (logits, cache).
        if "logits" in names and output:
            return output[0]
        if any(name in names for name in ("past_key_values", "cache_params", "past_key_value")) and len(output) > 1:
            return output[1]
    if isinstance(output, Mapping):
        for name in names:
            if name in output:
                return output[name]
    for name in names:
        value = getattr(output, name, None)
        if value is not None:
            return value
    return None


def _call_model(model: Any, kwargs: MutableMapping[str, Any]) -> Any:
    """Call a model while tolerating wrappers with a narrower signature."""
    callable_model = model if callable(model) else getattr(model, "forward", None)
    if callable(callable_model):
        try:
            return callable_model(**kwargs)
        except TypeError:
            # Some lightweight adapters do not accept generation-only kwargs.
            try:
                signature = inspect.signature(callable_model)
                accepted = {
                    key: value for key, value in kwargs.items() if key in signature.parameters
                }
                return callable_model(**accepted)
            except (TypeError, ValueError):
                raise

    generate = getattr(model, "generate", None) or getattr(model, "generate_tokens", None)
    if callable(generate):
        return generate(**kwargs)
    raise TypeError("Connected model is not callable and has no generate() method.")


def _sample_next_token(logits: Any, *, temperature: float, top_p: float, top_k: int,
                       decoding: str, generator: Any) -> Any:
    torch = _torch()
    logits = logits.float()
    if decoding == "greedy_fast" or float(temperature) <= 0.0:
        return logits.argmax(dim=-1, keepdim=True)

    temperature = max(float(temperature), 1e-5)
    logits = logits / temperature
    vocab_size = logits.shape[-1]
    if int(top_k) > 0 and int(top_k) < vocab_size:
        values, _ = torch.topk(logits, int(top_k), dim=-1)
        logits = logits.masked_fill(logits < values[..., -1, None], float("-inf"))
    top_p = min(max(float(top_p), 0.0), 1.0)
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf"))
        logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
    probabilities = torch.softmax(logits, dim=-1)
    return torch.multinomial(probabilities, num_samples=1, generator=generator)


def generate_tokens(
    model: Any,
    tokens: Any,
    attention_mask: Any = None,
    max_new_tokens: int = 512,
    temperature: float = 0.5,
    top_p: float = 1.0,
    top_k: int = 0,
    seed: int = 42,
    decoding: str = "greedy_fast",
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Any:
    """Generate token ids with one shared KV-cache-aware autoregressive loop.

    ``tokens`` may be an input-id tensor/list or a mapping containing
    ``input_ids``, ``inputs_embeds`` and additional model-specific values.  The
    returned tensor contains the original input ids followed by generated ids
    when input ids are available; for embedding-only inputs it contains only
    generated ids.  The model is never moved or cached by this function.
    """
    torch = _torch()
    generation_model = resolve_generation_model(model)
    if isinstance(tokens, Mapping):
        payload = dict(tokens)
    else:
        payload = {"input_ids": tokens}
    if "input_ids" not in payload and "inputs_embeds" not in payload:
        raise ValueError("Generation tokens must include input_ids or inputs_embeds.")

    input_ids = payload.get("input_ids")
    if input_ids is not None and not _is_tensor(input_ids):
        input_ids = torch.as_tensor(input_ids, dtype=torch.long)
        payload["input_ids"] = input_ids
    if input_ids is not None and input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
        payload["input_ids"] = input_ids
    if attention_mask is not None:
        payload["attention_mask"] = attention_mask
    if payload.get("attention_mask") is None:
        reference = input_ids if input_ids is not None else payload.get("inputs_embeds")
        if reference is not None:
            length = reference.shape[1]
            payload["attention_mask"] = torch.ones(
                (reference.shape[0], length), dtype=torch.long, device=reference.device
            )
    device = _model_device(generation_model) or _model_device(model)
    payload = _move_inputs_to_model_device(payload, device)
    input_ids = payload.get("input_ids")
    prompt_ids = input_ids.clone() if _is_tensor(input_ids) else None
    next_input = input_ids
    generated = []

    try:
        generator = torch.Generator(device=device) if device is not None else torch.Generator()
        generator.manual_seed(int(seed) & 0xFFFFFFFFFFFFFFFF)
    except (TypeError, RuntimeError):
        generator = torch.Generator()
        generator.manual_seed(int(seed) & 0xFFFFFFFFFFFFFFFF)

    tokenizer = resolve_tokenizer(model)
    eos_ids = set()
    for candidate in (tokenizer, generation_model, getattr(generation_model, "config", None)):
        if candidate is None:
            continue
        for name in ("eos_token_id", "eos_token_ids"):
            value = getattr(candidate, name, None)
            if value is None:
                continue
            if _is_tensor(value):
                value = value.reshape(-1).tolist()
            eos_ids.update(value if isinstance(value, (list, tuple, set)) else [value])
    past = None
    try:
        for step in range(max(0, int(max_new_tokens))):
            call_kwargs = dict(payload) if past is None else {
                "input_ids": next_input,
                "attention_mask": payload.get("attention_mask"),
                "past_key_values": past,
                "use_cache": True,
            }
            if past is None:
                call_kwargs["use_cache"] = True
            output = _call_model(generation_model, call_kwargs)
            logits = _get_output_value(output, "logits")
            if logits is None:
                # A backend may expose generate() only.  Delegate once and
                # return its result; this still shares parameter handling.
                generate = getattr(generation_model, "generate", None) or getattr(generation_model, "generate_tokens", None)
                if not callable(generate):
                    raise RuntimeError("The connected model returned no logits during generation.")
                fallback = dict(payload)
                fallback.update(
                    max_new_tokens=int(max_new_tokens),
                    do_sample=decoding != "greedy_fast",
                    temperature=float(temperature),
                    top_p=float(top_p),
                    top_k=int(top_k),
                    use_cache=True,
                )
                return generate(**fallback)

            next_token = _sample_next_token(
                logits[:, -1, :], temperature=temperature, top_p=top_p,
                top_k=top_k, decoding=decoding, generator=generator,
            )
            generated.append(next_token)
            if progress_callback is not None:
                progress_callback(step + 1, int(max_new_tokens))
            past = _get_output_value(output, "past_key_values", "cache_params", "past_key_value")
            if payload.get("attention_mask") is not None:
                ones = torch.ones(
                    (payload["attention_mask"].shape[0], 1),
                    dtype=payload["attention_mask"].dtype,
                    device=payload["attention_mask"].device,
                )
                payload["attention_mask"] = torch.cat((payload["attention_mask"], ones), dim=1)
            next_input = next_token
            if past is None and payload.get("input_ids") is not None and "inputs_embeds" not in payload:
                # Models without a cache still receive the complete context on
                # the next step, so sampling remains autoregressive.
                payload["input_ids"] = torch.cat((payload["input_ids"], next_token), dim=1)
            elif past is None and "inputs_embeds" in payload:
                # After an embedding-only prefill, fall back to token ids for
                # adapters that do not expose a KV cache. The next iteration
                # still receives the growing generated context.
                payload.pop("inputs_embeds", None)
                payload.pop("image_embeds", None)
                payload["input_ids"] = next_token
            if eos_ids and all(int(token) in eos_ids for token in next_token.reshape(-1).tolist()):
                break
    finally:
        # Drop references to the cache at the end of every execution.  Model
        # residency/offload remains entirely under ComfyUI's ModelPatcher.
        past = None
        next_input = None

    if not generated:
        return prompt_ids if prompt_ids is not None else torch.empty((1, 0), dtype=torch.long)
    generated_ids = torch.cat(generated, dim=1)
    return torch.cat((prompt_ids, generated_ids), dim=1) if prompt_ids is not None else generated_ids


def decode_tokens(model: Any, token_ids: Any, *, prompt_length: int = 0) -> str:
    """Decode generated ids using the connected model's tokenizer."""
    tokenizer = resolve_tokenizer(model)
    if tokenizer is None:
        for candidate in _unwrap_candidates(model):
            if callable(getattr(candidate, "batch_decode", None)) or callable(getattr(candidate, "decode", None)):
                tokenizer = candidate
                break
    generated = token_ids[:, int(prompt_length):] if getattr(token_ids, "ndim", 0) > 1 else token_ids[int(prompt_length):]
    if tokenizer is not None:
        for name in ("batch_decode", "decode"):
            decoder = getattr(tokenizer, name, None)
            if not callable(decoder):
                continue
            try:
                if name == "batch_decode":
                    result = decoder(generated, skip_special_tokens=True)
                    return str(result[0] if isinstance(result, (list, tuple)) else result).strip()
                return str(decoder(generated.tolist(), skip_special_tokens=True)).strip()
            except TypeError:
                try:
                    return str(decoder(generated.tolist())).strip()
                except Exception:
                    continue
            except Exception:
                continue
    return " ".join(str(item) for item in generated.reshape(-1).tolist()).strip()


__all__ = [
    "decode_tokens",
    "generate_tokens",
    "resolve_generation_model",
    "resolve_tokenizer",
]
