# LoomNest SLM fine-tune pipeline (shopping-assistant tool calling)

Fine-tune a small open model (default **Qwen2.5-1.5B-Instruct**, alternative 3B) with LoRA/QLoRA so it picks the right
`ShoppingAssistantTools` function, with the right arguments, in English, Tamil, Hindi and Tanglish; export to GGUF; serve with
Ollama. The service needs **no code change**: `LLM_MODEL=loomnest-shop-slm`.

## Honest status

| Step | Status |
|---|---|
| Tool-schema/prompt extraction from the Java sources | done and tested (24 tools, system prompt, language directive, policy texts) |
| Dataset generation + validation (`generate_dataset.py`, `validate_dataset.py`) | done, run, unit-tested |
| Chat-template fidelity (our renderer vs. the real Qwen2.5 tokenizer template) | checked: byte-identical on the sample; loss masking verified on it |
| `train_lora.py` `--help` / `--dry-run` | done, run (no ML libraries needed) |
| **Actual training, adapter merge, GGUF conversion, Ollama build, evaluation of the tuned model** | **NOT executed here** (no GPU). `train_lora.py`, `merge_adapter.py` and the commands below are written against the documented APIs and are unverified until you run them. Do a `--max-steps 20` smoke run first. |

The dataset is synthetic and template-based. The Tamil/Hindi/Tanglish wording is hand-written: have a native speaker review
`slm_lexicon.py` before relying on the tuned model in production. Expect the model to learn *behaviour* (which tool, which arguments, refuse
vs. call, reply language), not new product knowledge.

## Files

| File | Purpose |
|---|---|
| `java_tools.py` | string-aware scanner: `@Tool`/`@ToolParam` -> OpenAI tool schema; `SYSTEM_PROMPT`; `LanguageSupport` directive; policy texts |
| `slm_lexicon.py` | language packs, product vocabulary, utterance templates, sample catalogue |
| `generate_dataset.py` | builds `data/train.jsonl`, `val.jsonl`, `tools_schema.json`, `system_prompt.txt`, `dataset_stats.json` |
| `validate_dataset.py` | schema/grounding checks on every line |
| `train_lora.py` | QLoRA SFT (transformers + peft + trl), lazy imports |
| `merge_adapter.py` | merges the adapter into the base weights |
| `Modelfile` | Ollama model definition (official Qwen2.5 tool-calling template, temperature 0.3, stop tokens, 8k context) |
| `requirements.txt` | training environment only |
| `test_slm_tools.py` | unit tests (`python -m unittest test_slm_tools`) |

## 1. Dataset (CPU, stdlib only)

```powershell
cd Backend\ai-slm
python generate_dataset.py --n 3000 --seed 42          # deterministic; optional: --catalogue catalogue.json
python validate_dataset.py                              # exit 1 on any bad line
python train_lora.py --dry-run                          # checks files + prints the plan, no ML libs
python -m unittest test_slm_tools
```

Re-run `generate_dataset.py` whenever `ShoppingAssistantTools.java`, `ChatbotServiceImpl.SYSTEM_PROMPT` or `LanguageSupport` change, so
training matches the live tool schema. A tool with no template in `generate_dataset.py` is reported as "tools without examples".

Default run (3000 examples, seed 42; 2700 train / 300 val): examples per tool call - searchProducts 534, getStorePolicyInfo 221, draftReturnRequest 171,
getTrendingProducts 145, requestOrderReturn 133, trackOrder 120, getMyOrders 116, escalateToHumanSupport 108, askProductQuestion 105, getMyNotifications 103,
getMyCart 102, cancelOrder 99, getPersonalizedRecommendations 95, getProductSentiment 91, getSimilarProducts 85, addToWishlist 83, getProductReviews 82,
getProductReviewSummary 80, addToCart 78, getFrequentlyBoughtTogether 77, getRefundStatus 71, getUnreadNotificationCount 68, getMyWishlist 56, getMyProfile 52.
Languages: en 1198, hi 592, ta 592, tanglish 618. 419 no-tool examples (small talk, capability limits, order-id clarification, cross-user refusals, prompt-injection
refusals, "which tools do you have"), 304 multi-turn, 294 with 2+ tool calls (search -> similar/reviews/cart/compare, draft return -> confirm -> submit, return -> escalate).

Format (one JSON object per line; `tool_calls[].function.arguments` is an **object** because HF chat templates `tojson` it; pass `--arguments-as-string` for OpenAI
wire format, `--embed-tools` to copy the schema into each row):

```json
{"messages": [{"role": "system", "content": "<SYSTEM_PROMPT + language directive>"},
              {"role": "user", "content": "red saree 2000 kulla irukka?"},
              {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1a2b", "type": "function",
                 "function": {"name": "searchProducts", "arguments": {"keyword": "saree", "maxPrice": 2000}}}]},
              {"role": "tool", "tool_call_id": "call_1a2b", "name": "searchProducts", "content": "[{...}]"},
              {"role": "assistant", "content": "Idhu enakku kidaicha options:\n1. ..."}],
 "meta": {"kind": "search", "lang": "tanglish", "tools": ["searchProducts"]}}
```

What it teaches: canonical English catalogue keywords from Tamil/Hindi/Tanglish/typo'd input (`pencl` -> `pencil`), `maxPrice` from budgets, ids only from the conversation
(the validator rejects invented UUIDs), draft-before-submit returns, refusing cross-user requests and prompt injection *without* a tool call, "couldn't find" on unknown
products, replies in the user's script. Note the live server also blocks injections (HTTP 400) and masks PII before the model; those refusals are belt-and-braces.

## 2. Train (GPU)

Hardware: 1.5B QLoRA (4-bit) fits in roughly **8-12 GB VRAM**; a **free Colab T4 (16 GB)** works. 3B needs ~12-16 GB (use a larger Colab GPU).
Every example carries ~3.3k tokens of constant system prompt + 24 tool schemas (measured with the Qwen2.5 tokenizer: mean 3.3k, p95 3.7k, max 4.2k total), so
sequences are long: default `--max-seq-len 4608`, effective batch 16 (1 x 16 accumulation). If you hit OOM lower `--rank 8`, raise `--grad-accum`, or train on fewer
examples. Rough, unmeasured time estimate: 2-4 h on a T4 for 2 epochs of 3000 examples, well under 1 h on an A100/4090. Use `--epochs 1` or `--n 1500` to shorten.

```bash
# Colab (Runtime > T4 GPU) or any CUDA Linux box
git clone <repo> && cd <repo>/Backend/ai-slm        # or upload the folder
pip install -r requirements.txt
python generate_dataset.py --n 3000 && python validate_dataset.py
python train_lora.py --max-steps 20                 # smoke test first
python train_lora.py --epochs 2                     # or --base-model Qwen/Qwen2.5-3B-Instruct
# output: out/loomnest-shop-lora/adapter  (download it from Colab)
```

Options: `--base-model`, `--epochs`, `--lr` (2e-4), `--rank` (16; alpha = 2r), `--max-seq-len`, `--batch-size`, `--grad-accum`, `--no-4bit`, `--trainer {auto,sft,hf}`.
Loss covers assistant turns only (tool calls + final answers). Without CUDA the script exits with code 3 and prints Colab instructions. If the installed TRL version's
API differs, `--trainer auto` falls back to `transformers.Trainer` (same pre-tokenised data and masking).
The base license: Qwen2.5-1.5B is Apache-2.0; the 3B model uses the Qwen research license - check it before commercial use.

## 3. Merge, convert to GGUF, quantise

```bash
python merge_adapter.py --base-model Qwen/Qwen2.5-1.5B-Instruct --adapter out/loomnest-shop-lora/adapter --out out/merged

git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
pip install -r requirements/requirements-convert_hf_to_gguf.txt        # path may differ by version
python convert_hf_to_gguf.py ../out/merged --outfile ../out/loomnest-shop-slm.f16.gguf --outtype f16
cmake -B build && cmake --build build --config Release -t llama-quantize   # or download a release binary
./build/bin/llama-quantize ../out/loomnest-shop-slm.f16.gguf ../out/loomnest-shop-slm.Q4_K_M.gguf Q4_K_M
# Windows: build\bin\Release\llama-quantize.exe <in> <out> Q4_K_M
```

Alternative (unverified here): `ollama create loomnest-shop-slm -q q4_K_M -f Modelfile` with `FROM ./out/merged` (safetensors directory) lets Ollama convert and quantise itself.

## 4. Serve with Ollama

The `Modelfile` (next to this README) uses `FROM ./out/loomnest-shop-slm.Q4_K_M.gguf`, the Qwen2.5 tool-calling template (`<tool_call>` / `<tool_response>`),
`temperature 0.3`, stop tokens `<|im_start|>` / `<|im_end|>`, and `num_ctx 8192` (the default 2048 would truncate the system prompt + tool schemas). Compare its TEMPLATE with
`ollama show qwen2.5:1.5b --template`.

```bash
ollama create loomnest-shop-slm -f Modelfile
ollama run loomnest-shop-slm "hi"                       # sanity check
# then restart recommendation-service with:
#   LLM_MODEL=loomnest-shop-slm     (llm.model; llm.enabled=true and llm.base-url=http://localhost:11434/v1 must already be set, see recommendation-service/README.md)
```

Docker note: `docker-compose.yml`'s `ollama-init` runs `ollama pull ${LLM_MODEL}`, which fails for a locally built model. Use a host-installed Ollama, or run
`ollama create` inside the container (copy the GGUF + Modelfile in) and do not use `ollama-init` with this model name.
The embedding model (`LLM_EMBEDDING_MODEL`) is independent and unchanged.

## 5. Compare against the base model (Backend\ai-eval)

Use the same GPU/Ollama host for both runs. A fair baseline is the *untuned* `qwen2.5:1.5b`; also run the current default for reference.

```powershell
cd Backend\ai-eval        # tokens.json as described in ai-eval\README.md
# A) base: set LLM_MODEL=qwen2.5:1.5b, restart recommendation-service
python run_eval.py --tokens tokens.json --label qwen2.5-1.5b-base --out base.json
# B) tuned: set LLM_MODEL=loomnest-shop-slm, restart recommendation-service
python run_eval.py --tokens tokens.json --label loomnest-shop-slm --out slm.json --compare base.json --strict-categories guardrail,leak,rbac
```

Acceptance criteria (suggested):

1. Tool selection: pass rate of `shopper_tools`, `account`, `returns`, `policy` >= base, and, if the service exposes a tool trace (`toolsUsed`), `tool_check` pass >= 90%.
   Today the service exposes none, so `tool_check` is `unavailable` and tool selection is judged by the content assertions only.
2. No regression on `guardrail`, `leak`, `rbac`, `hallucination` (`--strict-categories` exit 0, no `REGRESSIONS` in those categories). The 400/403 guardrail cases are enforced
   server-side, so they act as a sanity check that the stack is wired correctly; the model-dependent ones are the prompt-leak, PII-echo and leak probes.
3. `multilingual` pass rate >= base (reply script matches the message script; Tanglish stays Latin).
4. Overall pass rate >= base and >= `--fail-under`; p95 latency not worse than base (on CPU the 1.5B model should be much faster than the 8B default).
5. Re-run any regressed case with `--id` a few times before judging (sampling noise at temperature 0.3).

If the tuned model fails (1)-(3), look at `report.json` failure reasons, extend the matching templates in `generate_dataset.py`, regenerate, retrain.

## Generated files

`data/train.jsonl` and `data/val.jsonl` are reproducible with the seed and are not meant to be committed; `out/` (adapters, merged weights, GGUF) is multi-GB - keep it out of git.



!git clone https://github.com/Ramesh-456/ai-slm.git
%cd ai-slm
!pip install -r requirements.txt
!python generate_dataset.py --n 3000 --seed 42 \
  --tools-java ShoppingAssistantTools.java \
  --chatbot-java ChatbotServiceImpl.java \
  --language-java LanguageSupport.java
!python validate_dataset.py
!python train_lora.py --max-steps 20





!pip install -r requirements.txt 
!python generate_dataset.py --n 3000 --seed 42 \
--tools-java ShoppingAssistantTools.java \
--chatbot-java ChatbotServiceImpl.java \
--language-java LanguageSupport.java 
!PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python validate_dataset.py 
!python train_lora.py --epochs 2 --trainer hf
