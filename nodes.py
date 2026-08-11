"""ToriiGate ComfyUI nodes.

This plugin deliberately has no model loader.  A native ComfyUI loader owns
the model, tokenizer, quantisation and ModelPatcher/offload lifecycle; these
nodes only build ToriiGate prompts and run generation on the connected object.
"""

from __future__ import annotations

from typing import Any, Mapping

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


def _call_variants(function, variants):
    """Call an adapter with a few deliberately small, compatible signatures."""
    last_error = None
    for args, kwargs in variants:
        try:
            return function(*args, **kwargs)
        except (TypeError, AttributeError) as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return None


def _image_to_pil(image, max_pixels_mp):
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("ToriiGate Reforged Caption requires Pillow from the ComfyUI runtime.") from exc
    if image is None:
        raise ValueError("ToriiGate Reforged Caption requires an IMAGE input.")
    try:
        first = image[0]
        array = first.detach().cpu().numpy() if hasattr(first, "detach") else first
    except (IndexError, TypeError, AttributeError) as exc:
        raise ValueError("IMAGE must be a ComfyUI image tensor with at least one image.") from exc
    try:
        import numpy as np
        array = np.asarray(array)
        if array.dtype.kind == "f":
            array = (array * 255.0).clip(0, 255).astype(np.uint8)
        else:
            array = array.clip(0, 255).astype(np.uint8)
    except ImportError as exc:
        raise RuntimeError("ToriiGate Reforged Caption requires NumPy from the ComfyUI runtime.") from exc
    pil = Image.fromarray(array)
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    limit = max(1, int(float(max_pixels_mp) * 1_000_000))
    pixels = pil.width * pil.height
    if pixels > limit:
        scale = (limit / pixels) ** 0.5
        pil = pil.resize((max(1, int(pil.width * scale)), max(1, int(pil.height * scale))), Image.Resampling.LANCZOS)
    return pil


def _render_messages(tokenizer, messages):
    apply_template = getattr(tokenizer, "apply_chat_template", None) if tokenizer is not None else None
    if callable(apply_template):
        try:
            return apply_template(messages, tokenize=False, add_generation_prompt=True)
        except (TypeError, ValueError):
            # Text-only tokenizers may reject the multimodal content list.
            text_messages = [
                {"role": msg["role"], "content": " ".join(
                    part.get("text", "") if isinstance(part, Mapping) else str(part)
                    for part in (msg.get("content", []) if isinstance(msg.get("content"), list) else [msg.get("content", "")])
                )}
                for msg in messages
            ]
            return apply_template(text_messages, tokenize=False, add_generation_prompt=True)
    pieces = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                part.get("text", "<image>") if isinstance(part, Mapping) else str(part)
                for part in content
            )
        pieces.append(f"{message.get('role', 'user')}: {content}")
    return "\n".join(pieces) + "\nassistant:"


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
            except (TypeError, ValueError, AttributeError):
                continue
    for candidate in (clip, resolve_generation_model(clip)):
        tokenize = getattr(candidate, "tokenize", None)
        if callable(tokenize):
            for args, kwargs in (((), {"text": rendered}), ((rendered,), {})):
                try:
                    return _normalise_token_inputs(tokenize(*args, **kwargs))
                except (TypeError, ValueError, AttributeError):
                    continue
    raise RuntimeError(
        "The connected CLIP/model has no usable tokenizer. "
        "Use a native loader that exposes the ToriiGate tokenizer."
    )


def _vision_candidates(vision):
    seen = set()
    pending = [vision]
    while pending:
        candidate = pending.pop(0)
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        yield candidate
        for name in ("model", "vision_model", "clip_vision", "mmproj", "projector"):
            child = getattr(candidate, name, None)
            if child is not None and child is not candidate:
                pending.append(child)


def _encode_vision(vision, image, source_image=None):
    if vision is None:
        raise RuntimeError(
            "ToriiGate Reforged Caption requires a connected vision/mmproj input. "
            "Load the compatible vision projector with ComfyUI and connect it to vision."
        )
    if isinstance(vision, Mapping):
        return dict(vision)
    if hasattr(vision, "shape"):
        return vision
    prepared = None
    for candidate in _vision_candidates(vision):
        for name in ("image_processor", "processor", "preprocess"):
            processor = getattr(candidate, name, None)
            if callable(processor):
                try:
                    prepared = processor(images=image, return_tensors="pt")
                except TypeError:
                    try:
                        prepared = processor(image)
                    except (TypeError, ValueError):
                        pass
                if prepared is not None:
                    break
        if prepared is not None:
            break
    for candidate in _vision_candidates(vision):
        for name in ("encode_image", "encode", "get_image_features", "forward"):
            encoder = getattr(candidate, name, None)
            if not callable(encoder):
                continue
            variants = []
            if prepared is not None:
                variants.extend([((prepared,), {}), ((), prepared if isinstance(prepared, Mapping) else {})])
            if source_image is not None:
                variants.append(((source_image,), {}))
            variants.extend([((image,), {}), ((), {"image": image}), ((), {"images": image})])
            try:
                output = _call_variants(encoder, variants)
            except (TypeError, AttributeError, ValueError):
                continue
            if output is not None:
                return output
    raise RuntimeError(
        "The connected vision/mmproj object cannot encode images. "
        "Connect a ToriiGate-compatible vision projector from ComfyUI's loader."
    )


def _prepare_multimodal_inputs(clip, vision, image, token_inputs, source_image=None):
    image_features = _encode_vision(vision, image, source_image=source_image)
    base = dict(token_inputs)
    generation_model = resolve_generation_model(clip)
    variants = []
    extra = {
        "input_ids": base.get("input_ids"),
        "attention_mask": base.get("attention_mask"),
        "image_embeds": image_features,
        "image_embeddings": image_features,
        "vision_embeddings": image_features,
        "image": image,
    }
    for candidate in (clip, generation_model, vision):
        for name in ("prepare_multimodal_inputs", "prepare_inputs_embeds", "build_multimodal_inputs", "insert_image_tokens", "encode_multimodal"):
            adapter = getattr(candidate, name, None)
            if not callable(adapter):
                continue
            variants.extend([
                ((), extra),
                ((base.get("input_ids"), image_features), {}),
                ((base, image_features), {}),
            ])
            try:
                result = _call_variants(adapter, variants)
            except (TypeError, AttributeError, ValueError):
                continue
            if isinstance(result, Mapping):
                merged = dict(result)
                if "attention_mask" not in merged and base.get("attention_mask") is not None:
                    merged["attention_mask"] = base["attention_mask"]
                return merged
            if result is not None:
                return {"inputs_embeds": result, "attention_mask": base.get("attention_mask")}

    # A loader may expose an embedding layer but no named adapter.  In that
    # case prepend projected image features to text embeddings.  This is the
    # generic representation consumed by the shared token loop; model-specific
    # loaders can still provide a richer adapter above.
    embedding_layer = getattr(generation_model, "get_input_embeddings", None)
    if callable(embedding_layer) and base.get("input_ids") is not None and hasattr(image_features, "shape"):
        try:
            text_embeddings = embedding_layer(base["input_ids"])
            if getattr(image_features, "ndim", 0) == 2:
                image_features = image_features.unsqueeze(0)
            import torch
            inputs_embeds = torch.cat((image_features, text_embeddings), dim=1)
            attention = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device)
            return {"inputs_embeds": inputs_embeds, "attention_mask": attention}
        except (RuntimeError, TypeError, AttributeError):
            pass

    # Preserve projector-specific values for a model whose forward method
    # accepts them directly.  This is still a standard model object contract;
    # no model identity or weight format is inspected.
    if isinstance(image_features, Mapping):
        base.update(image_features)
    else:
        base["image_embeds"] = image_features
    return base


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


def _decode_node_result(clip, output, input_ids):
    prompt_length = 0
    if input_ids is not None:
        try:
            prompt_length = int(input_ids.shape[1] if input_ids.ndim > 1 else input_ids.shape[0])
        except (AttributeError, IndexError):
            prompt_length = len(input_ids)
    # Embedding-based multimodal adapters return generated ids without the
    # original text ids. Do not strip the generated sequence in that case.
    try:
        output_length = int(output.shape[1] if output.ndim > 1 else output.shape[0])
        if output_length <= prompt_length:
            prompt_length = 0
    except (AttributeError, IndexError):
        pass
    return decode_tokens(clip, output, prompt_length=prompt_length)


class ToriiGateCaption:
    """Caption one image with a connected ComfyUI CLIP + vision/mmproj."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Image to caption; only the first image in a batch is used."}),
                "clip": ("CLIP", {"forceInput": True, "tooltip": "ToriiGate-compatible CLIP/model from a native ComfyUI loader. The node does not inspect its format or identity."}),
                "vision": ("CLIP_VISION", {"forceInput": True, "tooltip": "Vision/mmproj from the compatible native loader. Required for image captioning."}),
                "max_pixels_mp": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 8.0, "step": 0.1, "tooltip": "Resize budget in megapixels before the vision projector."}),
                "max_new_tokens": ("INT", {"default": 512, "min": 1, "max": 8192, "tooltip": "Maximum number of generated tokens."}),
                "temperature": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.01, "tooltip": "Sampling temperature; used when decoding is sample."}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Nucleus sampling threshold."}),
                "top_k": ("INT", {"default": 0, "min": 0, "max": 200, "step": 1, "tooltip": "Top-k sampling; 0 disables the limit."}),
                "decoding": (["greedy_fast", "sample"], {"default": "greedy_fast", "tooltip": "greedy_fast is deterministic; sample uses temperature/top-p/top-k."}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": "fixed", "tooltip": "Fixed seed for repeatable sampling. Randomization is controlled by ComfyUI's generation control."}),
            },
            "optional": {
                "prompt": ("STRING", {"multiline": True, "default": "", "forceInput": True, "tooltip": "ToriiGate prompt; connect ToriiGate Reforged Grounding Builder or leave blank for a short default."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("caption",)
    FUNCTION = "caption"
    CATEGORY = "ToriiGate"

    def caption(self, image, clip, vision=None, max_pixels_mp=1.0, max_new_tokens=512,
                temperature=0.5, top_p=1.0, top_k=0, decoding="greedy_fast",
                seed=42, prompt="", mmproj=None):
        if vision is None:
            vision = mmproj
        if vision is None:
            raise RuntimeError("ToriiGate Reforged Caption requires a vision/mmproj connection; the image projector is missing.")
        pil = _image_to_pil(image, max_pixels_mp)
        prompt = prompt.strip() if isinstance(prompt, str) else str(prompt or "")
        if not prompt:
            prompt = "Describe this image in detail."
        messages = [
            {"role": "system", "content": [{"type": "text", "text": TORIIGATE_SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]},
        ]
        token_inputs = _tokenize(clip, prompt, messages)
        multimodal = _prepare_multimodal_inputs(clip, vision, pil, token_inputs, source_image=image)
        callback = _progress_callback(max_new_tokens)
        output = generate_tokens(
            clip,
            multimodal,
            attention_mask=multimodal.get("attention_mask"),
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            seed=int(seed),
            decoding=decoding,
            progress_callback=callback,
        )
        return (_decode_node_result(clip, output, token_inputs.get("input_ids")),)


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


NODE_KEY_GROUNDING_BUILDER = "ToriiGate_Reforged_GroundingBuilder"
NODE_KEY_CAPTION = "ToriiGate_Reforged_Caption"
NODE_KEY_TEXT_GENERATE = "ToriiGate_Reforged_TextGenerate"

# Keep one canonical key per node. Legacy keys are intentionally not aliases:
# registering them would make old and new definitions collide in ComfyUI's
# node registry and would resurrect removed backends.
NODE_CLASS_MAPPINGS = {
    NODE_KEY_GROUNDING_BUILDER: ToriiGateGroundingBuilder,
    NODE_KEY_CAPTION: ToriiGateCaption,
    NODE_KEY_TEXT_GENERATE: ToriiGateTextGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_KEY_GROUNDING_BUILDER: "ToriiGate Reforged Grounding Builder",
    NODE_KEY_CAPTION: "ToriiGate Reforged Caption",
    NODE_KEY_TEXT_GENERATE: "ToriiGate Reforged Text Generate",
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
