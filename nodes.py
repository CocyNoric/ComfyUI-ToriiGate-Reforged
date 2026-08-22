"""ToriiGate ComfyUI nodes.

This plugin deliberately has no model loader.  A native ComfyUI loader owns
the model, tokenizer, quantisation and ModelPatcher/offload lifecycle; these
nodes only build ToriiGate prompts and run generation on the connected object.
"""

from __future__ import annotations

import inspect
import math
import re
from typing import Mapping

from .generation import decode_tokens, generate_tokens, resolve_generation_model, resolve_tokenizer
from .prompts import make_user_query, system_prompt as TORIIGATE_SYSTEM_PROMPT


CAPTION_TYPES = [
    "long_thoughts_v2",
    "long_thoughts",
    "json",
    "long",
    "min_structured_md",
    "json_comic",
    "md_comic",
    "min_structured_json",
    "chroma-style",
    "short",
]


def _empty_grounding():
    return {
        "tags": [],
        "characters": [],
        "char_p_tags": {"chars": {}, "skins": {}},
        "char_descr": {"chars": {}, "skins": {}},
    }


def _split_csv(value):
    if not value:
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


class ToriiGateGroundingBuilder:
    """Build the ToriiGate caption instruction from tags and character data."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "caption_type": (
                    CAPTION_TYPES,
                    {
                        "default": "short",
                        "tooltip": "Caption format. short is fastest; long is detailed natural text; json/min_structured produce structured output; long_thoughts_v2 is the most detailed.",
                    },
                ),
                "use_names": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Allows the model to use or try to recognize character names."},
                ),
                "add_tags": ("BOOLEAN", {"default": False, "tooltip": "Show and use general tags."}),
                "tags": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "General booru tags for the image, separated by commas. These are only added when add_tags is enabled.",
                    },
                ),
                "add_character_list": ("BOOLEAN", {"default": False, "tooltip": "Show and use character list."}),
                "character_names": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "General character list, separated by commas. Empty per-character names use this list by position.",
                    },
                ),
                "character_count": ("INT", {"default": 1, "min": 0, "max": 5, "step": 1, "tooltip": "Number of characters to configure."}),
                "add_character_tags": ("BOOLEAN", {"default": False, "tooltip": "Show and use character tags."}),
                "add_character_descriptions": ("BOOLEAN", {"default": False, "tooltip": "Show and use character descriptions."}),
                "char1_name": ("STRING", {"default": "", "tooltip": "Name/tag for character 1."}),
                "char1_tags": ("STRING", {"multiline": True, "default": "", "tooltip": "Booru tags for character 1, separated by commas."}),
                "char1_description": ("STRING", {"multiline": True, "default": "", "tooltip": "Free-form description for character 1."}),
                "char2_name": ("STRING", {"default": "", "tooltip": "Name/tag for character 2."}),
                "char2_tags": ("STRING", {"multiline": True, "default": "", "tooltip": "Booru tags for character 2, separated by commas."}),
                "char2_description": ("STRING", {"multiline": True, "default": "", "tooltip": "Free-form description for character 2."}),
                "char3_name": ("STRING", {"default": "", "tooltip": "Name/tag for character 3."}),
                "char3_tags": ("STRING", {"multiline": True, "default": "", "tooltip": "Booru tags for character 3, separated by commas."}),
                "char3_description": ("STRING", {"multiline": True, "default": "", "tooltip": "Free-form description for character 3."}),
                "char4_name": ("STRING", {"default": "", "tooltip": "Name/tag for character 4."}),
                "char4_tags": ("STRING", {"multiline": True, "default": "", "tooltip": "Booru tags for character 4, separated by commas."}),
                "char4_description": ("STRING", {"multiline": True, "default": "", "tooltip": "Free-form description for character 4."}),
                "char5_name": ("STRING", {"default": "", "tooltip": "Name/tag for character 5."}),
                "char5_tags": ("STRING", {"multiline": True, "default": "", "tooltip": "Booru tags for character 5, separated by commas."}),
                "char5_description": ("STRING", {"multiline": True, "default": "", "tooltip": "Free-form description for character 5."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "build_grounding"
    CATEGORY = "ToriiGate/Grounding"

    def build_grounding(
        self,
        caption_type="short",
        use_names=True,
        add_tags=False,
        tags="",
        add_character_list=False,
        character_names="",
        character_count=1,
        add_character_tags=False,
        add_character_descriptions=False,
        char1_name="",
        char1_tags="",
        char1_description="",
        char2_name="",
        char2_tags="",
        char2_description="",
        char3_name="",
        char3_tags="",
        char3_description="",
        char4_name="",
        char4_tags="",
        char4_description="",
        char5_name="",
        char5_tags="",
        char5_description="",
    ):
        item = _empty_grounding()
        if add_tags:
            item["tags"] = _split_csv(tags)
        if add_character_list:
            item["characters"] = _split_csv(character_names)

        char_entries = [
            (char1_name, char1_tags if add_character_tags else "", char1_description if add_character_descriptions else ""),
            (char2_name, char2_tags if add_character_tags else "", char2_description if add_character_descriptions else ""),
            (char3_name, char3_tags if add_character_tags else "", char3_description if add_character_descriptions else ""),
            (char4_name, char4_tags if add_character_tags else "", char4_description if add_character_descriptions else ""),
            (char5_name, char5_tags if add_character_tags else "", char5_description if add_character_descriptions else ""),
        ][: max(0, min(5, int(character_count)))]

        auto_chars = []
        for index, (raw_name, raw_tags, raw_description) in enumerate(char_entries):
            name = raw_name.strip() if raw_name else ""
            if not name and index < len(item["characters"]):
                name = item["characters"][index]
            if not name:
                continue
            auto_chars.append(name)
            parsed_tags = _split_csv(raw_tags)
            if parsed_tags:
                item["char_p_tags"]["chars"][name] = parsed_tags
            description = raw_description.strip() if raw_description else ""
            if description:
                item["char_descr"]["chars"][name] = description

        if auto_chars and not item["characters"]:
            item["characters"] = auto_chars

        return (make_user_query(
            item,
            c_type=caption_type,
            use_names=use_names,
            add_tags=add_tags,
            add_characters=add_character_list,
            add_char_tags=add_character_tags,
            add_description=add_character_descriptions,
            underscores_replace=False,
        ),)


_TORIIGATE_MIN_IMAGE_PIXELS = 256 * 32 * 32
_TORIIGATE_IMAGE_FACTOR = 32
_TORIIGATE_IMAGE_MEAN = (0.5, 0.5, 0.5)
_TORIIGATE_IMAGE_STD = (0.5, 0.5, 0.5)
# Current ComfyUI qwen_vl preprocessing defaults.  The local affine adapter
# below makes its normalization produce the official ToriiGate values without
# changing ComfyUI's global Qwen/VL behavior for other models.
_COMFY_QWEN_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
_COMFY_QWEN_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)


def _toriigate_image_dimensions(height, width, max_pixels):
    """Return ToriiGate dimensions aligned to the visual encoder's 32px grid."""
    factor = _TORIIGATE_IMAGE_FACTOR
    min_pixels = _TORIIGATE_MIN_IMAGE_PIXELS
    max_pixels = max(min_pixels, int(max_pixels))
    target_height = round(height / factor) * factor
    target_width = round(width / factor) * factor
    target_pixels = target_height * target_width
    if target_pixels > max_pixels:
        scale = math.sqrt((height * width) / max_pixels)
        target_height = max(factor, math.floor(height / scale / factor) * factor)
        target_width = max(factor, math.floor(width / scale / factor) * factor)
    elif target_pixels < min_pixels:
        scale = math.sqrt(min_pixels / (height * width))
        target_height = math.ceil(height * scale / factor) * factor
        target_width = math.ceil(width * scale / factor) * factor
    return target_height, target_width


def _resize_comfy_image(image, max_pixels_mp):
    """Return the first image at ToriiGate's final visual-encoder dimensions."""
    if image is None:
        raise ValueError("ToriiGate Caption Reforged requires an IMAGE input.")
    try:
        first = image[:1]
        if int(first.shape[0]) < 1:
            raise ValueError
        height = int(first.shape[1])
        width = int(first.shape[2])
        if height < 1 or width < 1:
            raise ValueError
    except (IndexError, TypeError, AttributeError, ValueError) as exc:
        raise ValueError("IMAGE must be a ComfyUI image tensor with at least one image.") from exc
    limit = max(1, int(float(max_pixels_mp) * 1_000_000))
    # Match Qwen's smart-resize grid here because ComfyUI's built-in Qwen3.5
    # processor uses a lower generic minimum and cannot enforce ToriiGate's
    # original 256*32*32 floor after its own 32-pixel rounding.
    target_height, target_width = _toriigate_image_dimensions(height, width, limit)
    if (height, width) != (target_height, target_width):
        try:
            import torch.nn.functional as functional
            channels_first = first.movedim(-1, 1)
            try:
                channels_first = functional.interpolate(
                    channels_first,
                    size=(target_height, target_width),
                    mode="bicubic",
                    align_corners=False,
                    antialias=True,
                )
            except TypeError:
                channels_first = functional.interpolate(
                    channels_first,
                    size=(target_height, target_width),
                    mode="bicubic",
                    align_corners=False,
                )
            first = channels_first.movedim(1, -1).clamp(0.0, 1.0)
        except (ImportError, AttributeError, RuntimeError, TypeError) as exc:
            raise RuntimeError("Could not resize the ComfyUI IMAGE tensor for ToriiGate captioning.") from exc
    return first


def _adapt_toriigate_image_preprocessing(image):
    """Adapt raw IMAGE values to the official ToriiGate normalization.

    ComfyUI's native Qwen visual path currently applies its CLIP mean/std
    internally.  Transforming the raw tensor first makes that later step
    mathematically equivalent to ``(pixel - 0.5) / 0.5`` for ToriiGate.
    """
    current_mean = image.new_tensor(_COMFY_QWEN_IMAGE_MEAN)
    current_std = image.new_tensor(_COMFY_QWEN_IMAGE_STD)
    official_mean = image.new_tensor(_TORIIGATE_IMAGE_MEAN)
    official_std = image.new_tensor(_TORIIGATE_IMAGE_STD)
    return (image - official_mean) / official_std * current_std + current_mean


def _message_content_text(content):
    if not isinstance(content, list):
        return str(content or "")
    pieces = []
    for part in content:
        if not isinstance(part, Mapping):
            pieces.append(str(part))
        elif part.get("type") == "image":
            # ComfyUI's Qwen3.5 tokenizer replaces image_pad with an embedded
            # image object supplied through clip.tokenize(..., image=image).
            pieces.append("<|vision_start|><|image_pad|><|vision_end|>")
        else:
            pieces.append(str(part.get("text", "")))
    return "".join(pieces)


def _render_messages(tokenizer, messages):
    apply_template = getattr(tokenizer, "apply_chat_template", None) if tokenizer is not None else None
    if callable(apply_template):
        # Prefer a tokenizer's own switch, but still normalize the rendered
        # suffix below because not every template accepts or honors it.
        for template_messages in (messages, [
            {"role": msg["role"], "content": " ".join(
                part.get("text", "") if isinstance(part, Mapping) else str(part)
                for part in (msg.get("content", []) if isinstance(msg.get("content"), list) else [msg.get("content", "")])
            )}
            for msg in messages
        ]):
            for extra_options in ({"enable_thinking": False}, {}):
                try:
                    rendered = apply_template(
                        template_messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        **extra_options,
                    )
                    return _ensure_thinking_disabled(rendered)
                except (TypeError, ValueError):
                    continue
    # ToriiGate is based on Qwen3.5.  ComfyUI exposes its native tokenizer
    # without apply_chat_template(), so render the equivalent ChatML directly.
    pieces = []
    for message in messages:
        role = message.get("role", "user")
        content = _message_content_text(message.get("content", ""))
        pieces.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    pieces.append("<|im_start|>assistant\n")
    return _ensure_thinking_disabled("".join(pieces))


_TRAILING_OPEN_THINK = re.compile(r"<think(?:\s[^>]*)?>\s*\Z", re.IGNORECASE)
_TRAILING_CLOSED_THINK = re.compile(r"</think>\s*\Z", re.IGNORECASE)
_EMPTY_THINK_BLOCK = "<think>\n\n</think>\n"


def _ensure_thinking_disabled(rendered):
    """Prefill ToriiGate's empty reasoning block before generation starts."""
    rendered = str(rendered or "")
    if _TRAILING_CLOSED_THINK.search(rendered):
        return rendered
    open_think = _TRAILING_OPEN_THINK.search(rendered)
    if open_think:
        return rendered[:open_think.start()] + _EMPTY_THINK_BLOCK
    return rendered + _EMPTY_THINK_BLOCK


_LEADING_THINK_BLOCK = re.compile(r"\A\s*<think(?:\s[^>]*)?>.*?</think>\s*", re.IGNORECASE | re.DOTALL)


def _clean_generated_text(text):
    """Remove Qwen's private leading reasoning block from user-facing output."""
    cleaned = str(text or "").strip()
    while True:
        updated = _LEADING_THINK_BLOCK.sub("", cleaned, count=1).lstrip()
        if updated == cleaned:
            return cleaned
        cleaned = updated


def _normalise_token_inputs(value):
    if isinstance(value, Mapping):
        result = dict(value)
    else:
        result = {"input_ids": value}
    if "input_ids" not in result:
        raise RuntimeError("The connected tokenizer did not return input_ids.")
    return result


def _tokenize(clip, text, messages, use_template=True):
    tokenizer = resolve_tokenizer(clip)
    rendered = _render_messages(tokenizer, messages) if use_template else text
    if tokenizer is not None and callable(tokenizer):
        for kwargs in (
            {"return_tensors": "pt", "add_special_tokens": False},
            {"return_tensors": "pt"},
        ):
            try:
                return _normalise_token_inputs(tokenizer(rendered, **kwargs))
            except (TypeError, ValueError, AttributeError, RuntimeError):
                continue
    for candidate in (clip, resolve_generation_model(clip)):
        tokenize = getattr(candidate, "tokenize", None)
        if callable(tokenize):
            for args, kwargs in (((), {"text": rendered}), ((rendered,), {})):
                try:
                    return _normalise_token_inputs(tokenize(*args, **kwargs))
                except (TypeError, ValueError, AttributeError, RuntimeError):
                    continue
    raise RuntimeError(
        "The connected CLIP/model has no usable tokenizer. "
        "Use a native loader that exposes the ToriiGate tokenizer."
    )


def _supports_native_generation(clip):
    return all(callable(getattr(clip, name, None)) for name in ("tokenize", "generate", "decode"))


def _contains_image_token(value, seen=None):
    if seen is None:
        seen = set()
    if value is None or id(value) in seen:
        return False
    seen.add(id(value))
    if isinstance(value, Mapping):
        if value.get("type") == "image":
            return True
        if any(key in value for key in ("pixel_values", "image_embeds", "image_embeddings")):
            return True
        return any(_contains_image_token(item, seen) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_contains_image_token(item, seen) for item in value)
    return False


def _supported_kwargs(function, kwargs):
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return kwargs
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def _native_generate_text(clip, rendered, *, image=None, raw=False, max_new_tokens=512,
                          temperature=0.5, top_p=1.0, top_k=0,
                          decoding="greedy_fast", seed=42):
    """Use ComfyUI's CLIP-native tokenizer/generator, including embedded images."""
    if not _supports_native_generation(clip):
        raise RuntimeError(
            "The connected CLIP does not expose ComfyUI's native tokenize/generate/decode interface. "
            "Load ToriiGate with a compatible native CLIP loader."
        )

    tokenize = clip.tokenize
    tokenizer_options = {"thinking": False}
    if raw:
        # ComfyUI's Qwen tokenizer otherwise applies its user/assistant
        # template even when Text Generate is explicitly in raw mode.
        tokenizer_options = {"llama_template": "{}", "thinking": True}
    if image is None:
        try:
            tokens = tokenize(rendered, **tokenizer_options)
        except TypeError:
            tokens = tokenize(rendered)
    else:
        tokenization_errors = []
        tokens = None
        for image_kwargs in ({"image": image}, {"images": [image]}):
            try:
                tokens = tokenize(rendered, **image_kwargs, **tokenizer_options)
                break
            except TypeError as error:
                tokenization_errors.append(error)
                try:
                    tokens = tokenize(rendered, **image_kwargs)
                    break
                except TypeError as fallback_error:
                    tokenization_errors.append(fallback_error)
        if tokens is None:
            raise RuntimeError(
                "The connected CLIP tokenizer cannot accept ComfyUI IMAGE data. "
                "Load the ToriiGate/Qwen3.5 CLIP model with its native model type."
            ) from tokenization_errors[-1]
        if not _contains_image_token(tokens):
            raise RuntimeError(
                "The connected CLIP tokenizer ignored the IMAGE input. "
                "Load a ToriiGate/Qwen3.5 CLIP model that includes its built-in visual encoder."
            )

    generate = clip.generate
    kwargs = _supported_kwargs(generate, {
        "do_sample": decoding == "sample" and float(temperature) > 0.0,
        "max_length": int(max_new_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "top_k": int(top_k),
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
        "seed": int(seed),
    })
    output = generate(tokens, **kwargs)
    if hasattr(output, "tolist"):
        output = output.tolist()
    if isinstance(output, list) and len(output) == 1 and isinstance(output[0], list):
        output = output[0]
    try:
        decoded = clip.decode(output, skip_special_tokens=True)
    except TypeError:
        decoded = clip.decode(output)
    return _clean_generated_text(decoded)


def _progress_callback(max_new_tokens):
    try:
        from comfy.utils import ProgressBar
        progress = ProgressBar(int(max_new_tokens))
    except (ImportError, AttributeError, TypeError):
        return None

    def update(current, total):
        if current == total or current == 1 or current % 8 == 0:
            try:
                progress.update_absolute(current, total)
            except AttributeError:
                try:
                    progress.update(1)
                except Exception:
                    pass
    return update


def _token_sequence_length(value):
    if value is None:
        return 0
    try:
        return int(value.shape[1] if value.ndim > 1 else value.shape[0])
    except (AttributeError, IndexError, TypeError):
        pass
    try:
        first = value[0]
        if isinstance(first, (list, tuple)):
            return len(first)
        return len(value)
    except (IndexError, TypeError):
        return 0


def _decode_node_result(clip, output, input_ids):
    prompt_length = _token_sequence_length(input_ids)
    # Embedding-based multimodal adapters return generated ids without the
    # original text ids. Do not strip the generated sequence in that case.
    output_length = _token_sequence_length(output)
    if output_length <= prompt_length:
        prompt_length = 0
    return _clean_generated_text(decode_tokens(clip, output, prompt_length=prompt_length))


class ToriiGateCaption:
    """Caption one image with a native ComfyUI ToriiGate CLIP model."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {"forceInput": True, "tooltip": "ToriiGate-compatible CLIP/model from a native ComfyUI loader. The node does not inspect its format or identity."}),
                "image": ("IMAGE", {"tooltip": "Image to caption; only the first image in the batch is used."}),
                "max_pixels_mp": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 8.0, "step": 0.1, "tooltip": "Maximum resize budget before the built-in visual encoder. ToriiGate's original ~0.262 MP minimum is preserved, so 256x256 is enlarged to 512x512."}),
                "max_new_tokens": ("INT", {"default": 512, "min": 1, "max": 8192, "tooltip": "Maximum number of generated tokens."}),
                "temperature": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.01, "tooltip": "Sampling temperature; used when decoding is sample."}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Nucleus sampling threshold."}),
                "top_k": ("INT", {"default": 0, "min": 0, "max": 200, "step": 1, "tooltip": "Top-k sampling; 0 disables the limit."}),
                "decoding": (["greedy_fast", "sample"], {"default": "greedy_fast", "tooltip": "greedy_fast is deterministic; sample uses temperature/top-p/top-k."}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": "fixed", "tooltip": "Fixed seed for repeatable sampling. Randomization is controlled by ComfyUI's generation control."}),
            },
            "optional": {
                "prompt": ("STRING", {"multiline": True, "default": "", "forceInput": True, "tooltip": "ToriiGate prompt; connect ToriiGate Grounding Builder Reforged or leave blank for a short default."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("caption",)
    FUNCTION = "caption"
    CATEGORY = "ToriiGate"

    def caption(self, clip, image, max_pixels_mp=1.0, max_new_tokens=512,
                temperature=0.5, top_p=1.0, top_k=0, decoding="greedy_fast",
                seed=42, prompt=""):
        model_image = _adapt_toriigate_image_preprocessing(
            _resize_comfy_image(image, max_pixels_mp)
        )
        prompt = prompt.strip() if isinstance(prompt, str) else str(prompt or "")
        if not prompt:
            prompt = "Describe this image in detail."
        messages = [
            {"role": "system", "content": [{"type": "text", "text": TORIIGATE_SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]},
        ]
        rendered = _render_messages(resolve_tokenizer(clip), messages)
        return (_native_generate_text(
            clip,
            rendered,
            image=model_image,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            decoding=decoding,
            seed=seed,
        ),)


class ToriiGateTextGenerate:
    """Text-only ToriiGate generation using the same sampler as Caption."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {"forceInput": True, "tooltip": "ToriiGate-compatible model from a native ComfyUI loader."}),
                "prompt": ("STRING", {"multiline": True, "default": "", "forceInput": True, "tooltip": "Text prompt to generate from."}),
                "max_new_tokens": ("INT", {"default": 512, "min": 1, "max": 8192, "tooltip": "Maximum number of generated tokens."}),
                "temperature": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.01, "tooltip": "Sampling temperature; used when decoding is sample."}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Nucleus sampling threshold."}),
                "top_k": ("INT", {"default": 0, "min": 0, "max": 200, "step": 1, "tooltip": "Top-k sampling; 0 disables the limit."}),
                "decoding": (["greedy_fast", "sample"], {"default": "greedy_fast", "tooltip": "greedy_fast is deterministic; sample uses temperature/top-p/top-k."}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": "fixed", "tooltip": "Fixed seed for repeatable sampling."}),
            },
            "optional": {
                "system_prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "Optional system instruction."}),
                "template_mode": (["toriigate", "raw"], {"default": "toriigate", "tooltip": "toriigate applies the model's chat template; raw tokenizes the supplied text directly."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate"
    CATEGORY = "ToriiGate"

    def generate(self, clip, prompt="", max_new_tokens=512, temperature=0.5,
                 top_p=1.0, top_k=0, decoding="greedy_fast", seed=42,
                 system_prompt="", template_mode="toriigate"):
        prompt = str(prompt or "")
        if template_mode == "raw":
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = []
            system_text = str(system_prompt or "").strip() or TORIIGATE_SYSTEM_PROMPT
            if system_text:
                messages.append({"role": "system", "content": system_text})
            messages.append({"role": "user", "content": prompt})
        if _supports_native_generation(clip):
            rendered = prompt if template_mode == "raw" else _render_messages(resolve_tokenizer(clip), messages)
            return (_native_generate_text(
                clip,
                rendered,
                raw=template_mode == "raw",
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                decoding=decoding,
                seed=seed,
            ),)
        token_inputs = _tokenize(clip, prompt, messages, use_template=template_mode != "raw")
        output = generate_tokens(
            clip,
            token_inputs,
            attention_mask=token_inputs.get("attention_mask"),
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            seed=int(seed),
            decoding=decoding,
            progress_callback=_progress_callback(max_new_tokens),
        )
        return (_decode_node_result(clip, output, token_inputs.get("input_ids")),)


NODE_KEY_GROUNDING_BUILDER = "ToriiGate_GroundingBuilder_Reforged"
NODE_KEY_CAPTION = "ToriiGate_Caption_Reforged"
NODE_KEY_TEXT_GENERATE = "ToriiGate_TextGenerate_Reforged"

# Keep one canonical key per node. Legacy keys are intentionally not aliases:
# registering them would make old and new definitions collide in ComfyUI's
# node registry and would resurrect removed backends.
NODE_CLASS_MAPPINGS = {
    NODE_KEY_GROUNDING_BUILDER: ToriiGateGroundingBuilder,
    NODE_KEY_CAPTION: ToriiGateCaption,
    NODE_KEY_TEXT_GENERATE: ToriiGateTextGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_KEY_GROUNDING_BUILDER: "ToriiGate Grounding Builder Reforged",
    NODE_KEY_CAPTION: "ToriiGate Caption Reforged",
    NODE_KEY_TEXT_GENERATE: "ToriiGate Text Generate Reforged",
}

if set(NODE_CLASS_MAPPINGS) != set(NODE_DISPLAY_NAME_MAPPINGS):
    raise RuntimeError("ToriiGate node class/display registrations are out of sync")


__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "NODE_KEY_CAPTION",
    "NODE_KEY_GROUNDING_BUILDER",
    "NODE_KEY_TEXT_GENERATE",
    "ToriiGateCaption",
    "ToriiGateGroundingBuilder",
    "ToriiGateTextGenerate",
]
