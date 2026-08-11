# ComfyUI ToriiGate Reforged

ToriiGate nodes for ComfyUI. The plugin owns only ToriiGate prompt formatting,
vision captioning and text generation. Model loading, quantisation, device
placement and VRAM offload remain entirely under ComfyUI.

## Important: breaking migration

This release is a breaking replacement for the earlier loader workflows.
Existing legacy node graphs are not compatible. Remove the old nodes and
reconnect a native ComfyUI loader.

All registered node IDs use the `ToriiGate_Reforged_` prefix so this package
can coexist with an older ToriiGate installation without registry collisions.

## Local workflow

1. Prepare a ToriiGate-compatible INT8ConvRot or BF16 model yourself. A
   compatible GGUF is also fine when the installed native loader supports it.
2. Prepare the matching vision/mmproj projector.
3. Load both with the ComfyUI-native loader(s) for your installation.
4. Connect the resulting `CLIP`/model and `CLIP_VISION`/vision outputs to
   `ToriiGate Reforged Caption`, or connect only the model to
   `ToriiGate Reforged Text Generate`.
5. Connect `ToriiGate Reforged Grounding Builder` to the Caption `prompt` input
   when grounding tags, character names, tags or descriptions are needed.

No external server, network endpoint, repository ID, automatic download or
additional model package is required. The loader decides whether the weights
run as INT8ConvRot, BF16 or GGUF and ComfyUI decides when to keep or offload
them. The same model may safely feed Caption and Text Generate in one workflow.

## Nodes

### ToriiGate Reforged Grounding Builder

Builds the ToriiGate prompt template while preserving the existing Caption
type, character names, general tags, per-character tags and descriptions. It
does not load or inspect a model.

### ToriiGate Reforged Caption

Inputs are an `IMAGE`, a native `CLIP`/model, a compatible vision/mmproj, an
optional prompt, pixel budget, generation length, temperature, top-p, top-k,
decoding mode and seed. The node performs image preprocessing, vision
projection, image-token insertion, ToriiGate chat templating and generation,
then returns a `STRING` caption.

The node does not check model names, metadata, hashes or weight formats. If an
incompatible model is connected, ComfyUI reports the model/adapter error; the
tooltip and error identify that a ToriiGate-compatible model and mmproj are
required.

`greedy_fast` is deterministic. `sample` uses temperature/top-p/top-k and the
seed. Seed defaults to **42** and is fixed; seed `0` is a normal deterministic
seed value rather than a special random sentinel. ComfyUI automatic progress
feedback is used when its progress hook is available; there is no user
progress toggle.

### ToriiGate Reforged Text Generate

Text-only generation using the exact same tokenizer, sampler and KV-cache loop
as Caption. Inputs are the native model, `system_prompt`, `prompt`, generation
controls and `template_mode` (`toriigate` or `raw`). ToriiGate is primarily an
image-captioning model, so general chat quality is not guaranteed.

## Memory and execution

The nodes never call `.to("cuda")`, move a model to CPU, empty CUDA caches,
download weights, or retain a private model cache. They release local
generation/KV-cache references after each execution and leave residency and
offload decisions to ComfyUI's ModelPatcher. This lets a single loaded model
be reused by Caption, Text Generate and downstream diffusion nodes.

## Validation checklist

The intended workflow should be checked with INT8ConvRot and BF16 image
captioning, text-only generation, one model connected to both nodes, a Caption
followed by diffusion (to exercise ComfyUI offload), repeated executions,
missing-mmproj errors, fixed-seed sampling and long generations that release
their KV cache on completion.
