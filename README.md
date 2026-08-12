# ComfyUI-ToriiGate-Reforged

> A ComfyUI custom node pack for image captioning and text generation with ToriiGate-0.5, a fine-tuned Qwen3.5-4B model.

[中文说明](README_zh.md)

This is a Reforged version of [litch230/comfyui_toriigate](https://github.com/litch230/comfyui_toriigate). It removes the bundled Transformers loader and llama.cpp API nodes, and uses ComfyUI's native CLIP/Qwen3.5 loading, image understanding and generation instead.

## Changes

- Removed the custom model loader, automatic download and llama.cpp backend.
- Uses ComfyUI's native `Load CLIP` node and model/offload management.
- Uses the visual encoder included in the ToriiGate/Qwen3.5 CLIP model; no separate vision input is required.
- Keeps the original ToriiGate prompt formats, disables thinking before generation, and aligns the original image-size and visual-normalization behavior.

Old workflows must replace the legacy ToriiGate nodes with the Reforged nodes.

## Installation

Clone this repository into `ComfyUI/custom_nodes/`, then restart ComfyUI:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/CocyNoric/ComfyUI-ToriiGate-Reforged.git
```

## Model

Recommended model downloads:

- [Hugging Face: Ronysoc/ToriiGate-0.5-Int8-ConvRot](https://huggingface.co/Ronysoc/ToriiGate-0.5-Int8-ConvRot)
- [ModelScope: CocyNoric/ToriiGate-0.5-Int8-ConvRot](https://modelscope.cn/models/CocyNoric/ToriiGate-0.5-Int8-ConvRot) — recommended for users in China

Alternatively, obtain and merge the weights from the [original ToriiGate repository](https://huggingface.co/Minthy/ToriiGate-0.5).

Put a ComfyUI-compatible ToriiGate `.safetensors` weight, such as `ToriiGate-0.5_int8_convrot.safetensors`, in:

```text
ComfyUI/models/clip/
```

In `Load CLIP` select:

- `clip_name`: your ToriiGate weight
- `type`: `stable_diffusion`
- `device`: `default`

## Usage and connections

![Workflow connections](assets/show.png)

You can drag [the example workflow](assets/ToriiGate%20Inferencer%20Reforged.json) into ComfyUI, or connect the nodes manually:

```text
Load CLIP.CLIP ────────────────> ToriiGate Caption Reforged.clip
Load Image.IMAGE ──────────────> ToriiGate Caption Reforged.image
Grounding Builder.prompt ──────> ToriiGate Caption Reforged.prompt (optional)
ToriiGate Caption.caption ─────> Preview Any.source
```

For text-only generation, connect `Load CLIP.CLIP` to `ToriiGate Text Generate Reforged.clip`, then provide a prompt.

## Nodes

### ToriiGate Grounding Builder Reforged

Builds ToriiGate prompts and optional tags/character information.

![ToriiGate Grounding Builder Reforged](assets/show_node2.png)

#### Caption types

`caption_type` controls the output format and detail level:

> **Tip:** The `long` mode is recommended for general use.

> **Warning:** Testing found that character identification accuracy may not always be ideal, with either the quantized or original weights.

| Mode | Output |
| --- | --- |
| `short` | Short caption covering the main subjects and details. |
| `long` | Detailed natural-language caption in 2–5 paragraphs. |
| `long_thoughts` | Structured character analysis, general description, individual parts, visible text, background and effects. |
| `long_thoughts_v2` | Detailed character identification, key details, long description and per-character descriptions. |
| `json` | JSON-style description of characters/main content, background, effects, text and atmosphere. |
| `min_structured_md` | Shorter Markdown structure for key details, general content, characters and image effects. |
| `min_structured_json` | Compact JSON-style general, character, effect, text and watermark fields. |
| `json_comic` | Comic-focused JSON with frame descriptions, characters and overall meaning. |
| `md_comic` | Comic-focused Markdown with layout and per-frame descriptions. |
| `chroma-style` | Four views: regular summary, item list, Midjourney-style summary and commission description. |

### ToriiGate Caption Reforged

Generates a caption from the connected CLIP model, image and optional grounding prompt.

![ToriiGate Caption Reforged](assets/show_node1.png)

### ToriiGate Text Generate Reforged

Performs text-only generation with the connected CLIP model.

![ToriiGate Text Generate Reforged](assets/show_node3.png)

## Acknowledgements

- Original ComfyUI extension: [litch230/comfyui_toriigate](https://github.com/litch230/comfyui_toriigate)
- ToriiGate model, prompt formats and inference reference: [Minthy/ToriiGate-0.5](https://huggingface.co/Minthy/ToriiGate-0.5)
