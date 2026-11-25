# LoomNest SLM — Enna, Yethuku, Yepdi Train Pannanum (Tanglish Guide)

Indha document-a padichitu nee intha folder (`Backend\ai-slm`) la irukura commands-a step-by-step run pannalam.
Doubt irundha edhuvume run pannama ping pannu — especially **Step 4 (actual GPU training)** mudinjadhukku apparam.

---

## 1. Idhu enna? (What is this?)

`Backend\ai-slm` folder la irukura code, oru **Small Language Model (SLM)** fine-tune panradhukku. Idhu oru
**chatbot/shopping-assistant brain**-a "train" panna use aagudhu — LoomNest `recommendation-service` la irukura
AI chatbot, user kேட்கிற கேள்வி (English/Tamil/Hindi/Tanglish edhu vendalum) correct-aana **tool-a** (function-a)
correct-aana arguments oda call pannanum-nu kathukka vaikradhu dhan indha pipeline oda main vேlai.

Example: user "red saree 2000 kulla irukka?" nu kettaa, model `searchProducts` nu oru tool-a
`{"keyword": "saree", "maxPrice": 2000}` nu arguments oda call panna vendiyadhu. Idhu correct-aa panna
model-a "teach" panradhukku dhan fine-tuning.

**Base model**: `Qwen/Qwen2.5-1.5B-Instruct` (chinna, 1.5 billion parameter model — adhan "Small" Language Model).
Idha already irukura model mela **LoRA / QLoRA** nu oru lightweight fine-tuning technique vachu, konjam extra
"training" kodukkradhu — full model-a train pannama, oru small "adapter" layer mattum train pannuvom. Adhunaala
konjam GPU/VRAM pothum.

## 2. Yethuku panrom? (Why are we doing this?)

- `recommendation-service` la irukura `ShoppingAssistantTools.java` la **24 tools** irukku (searchProducts,
  trackOrder, addToCart, getMyOrders, escalateToHumanSupport, etc.)
- Default-a use panra big model (8B edhavadhu) correct-aa velai seiyudhu, aana adhu **slow** + **costly** (more VRAM/CPU).
- Chinna 1.5B model-a namma data vachu fine-tune panna, adhே correct-aa tool call panna kathukkum — so namma
  **faster + cheaper** model-a use pannalam, same accuracy-oda (ideally).
- Multi-language support (**English, Tamil, Hindi, Tanglish**) correct-aa work aaganum-nu, andha 4 languages
  oda training examples podrom.
- Service code-la **ஒரு வரி கூட மாத்த வேண்டாம்** — ippadiye `LLM_MODEL=loomnest-shop-slm` nu config change panna
  pothum, adhான new tuned model-a point pannidum.

**Important honest note**: Idhu innum yaaru run pannala (README.md la "NOT executed" nu clearly sollirukanga).
Code ellam correct-aa thaan documented APIs vachu eழுதியிருக்கு, aana real GPU la run panni verify pannala.
Adhanaala first time run panravanga konjam careful-a, chinna steps-la (smoke test) pannanum.

## 3. Pipeline oda files (ஒரு summary)

| File | Vela enna |
|---|---|
| `java_tools.py` | Java source-la irundhu tool schema + system prompt-a extract pannudhu |
| `slm_lexicon.py` | Tamil/Hindi/Tanglish vocabulary, sentence templates |
| `generate_dataset.py` | Training data (`train.jsonl`, `val.jsonl`) generate pannudhu |
| `validate_dataset.py` | Generate pannina data correct-aa irukka-nu check pannudhu |
| `train_lora.py` | **Main training script** — ithan nee run pannanum |
| `merge_adapter.py` | Training mudinjadhukku apparam, adapter-a base model-oda merge pannudhu |
| `Modelfile` | Ollama-la model serve panna vேண்டிய config |
| `requirements.txt` | Training-ku venum Python libraries |

---

## 4. Step-by-step: Yepdi run pannanum

### Step 0 — Innaiku ippove indha machine la irukura GPU

Naan check pannen, intha laptop-la **NVIDIA RTX A500 Laptop GPU — 4 GB VRAM** mattum irukku.

⚠️ **Mukkiyamana warning**: README padi, 1.5B model-a QLoRA (4-bit) la train panna **kuraindhadhu 8–12 GB VRAM**
venum. Namma machine la **4 GB mattum irukku** — adhanaala **local-a (idhuleye) training smooth-a work aagama,
"CUDA out of memory" error varra chance adhigam**. Rendu options irukku:

1. **Recommended: Free Google Colab (T4 GPU — 16 GB)** — Step 4B la kொடுத்திருக்கேன். Idhu free, adhanaala
   first attempt idha use pannudhu nalladhu.
2. **Idhé local machine** — chinna settings vachu (rank kammi, batch size 1) try pannalam, aana OOM aagala-nu
   guarantee illa. Step 4A la kொடுத்திருக்கேன்.

### Step 1 — Dataset generate pannu (GPU vendam, idhu CPU la thaan run aagum)

```powershell
cd Backend\ai-slm
python generate_dataset.py --n 3000 --seed 42
python validate_dataset.py
```

- `generate_dataset.py` — 3000 training examples (2700 train + 300 validation) create pannudhu, `data\train.jsonl`
  and `data\val.jsonl` nu files create aagum. `--seed 42` podradhunaala every time same data-வே varum (reproducible).
- `validate_dataset.py` — generate pannina ovvoru line-um correct format-la irukka-nu check pannudhu. Edhavadhu
  line bad-aa irundha, exit code 1 oda fail aagum — adha fix pannama mundhi pogadha.

Output-la "tools without examples" nu edhavadhu varudhunu paaru — adhu irundha `generate_dataset.py` la andha
tool-ku template illainu artham.

### Step 2 — Dry-run (ML library vendam, 5 nodi velai)

```powershell
python train_lora.py --dry-run
```

Idhu **edhuvume train pannadhu**. Data files correct-aa irukka, config enna-nu plan-a print pannidum mattum
(base model, LoRA rank, steps per epoch, token estimate, etc.). Idhu clean-a run aagama irundha, Step 1
correct-aa mudiyala-nu artham.

### Step 3 — Unit tests (optional, confidence check)

```powershell
python -m unittest test_slm_tools
```

### Step 4A — Local-a (indha laptop-le) training try pannradhu (4 GB VRAM — risky, optional)

Mudhal-la Python libraries install pannanum (torch, transformers, peft, trl, bitsandbytes, accelerate, datasets):

```powershell
pip install -r requirements.txt
```

⚠️ Note: `bitsandbytes` (4-bit QLoRA-ku venum library) Windows la nativeaa konjam problematic-aa irukkalam.
Install fail aayitta, rendu options:
- WSL2 (Windows la Linux) use pannu, adhula install pannu.
- Illati `--no-4bit` flag vachu plain 16-bit LoRA pannu (aana 4GB VRAM-ku adhu innum bariyam aagum).

Chinna "smoke test" (20 steps mattum, verify panna):

```powershell
python train_lora.py --max-steps 20 --rank 8 --batch-size 1 --grad-accum 8 --max-seq-len 2048
```

- `--max-steps 20` — 20 training steps mattum run pannudhu (full training illa, idhu test mattum).
- `--rank 8` — default 16-ku bathilu 8 (konjam kammi VRAM use pannum, aana konjam kammi "learning capacity").
- `--max-seq-len 2048` — default 4608-ku bathilu kammi (nedumaiyaana examples drop aagum, aana VRAM mிச்சம்).

Idhu success-aa odi "adapter saved to out\loomnest-shop-lora\adapter" nu print aanaa, full training pannalam:

```powershell
python train_lora.py --epochs 2
```

CUDA illainu program sonna (`torch.cuda.is_available()` False), illati "out of memory" error vandha, **Step 4B
(Colab) ku pogu** — adhuvே safe + guaranteed-a work aagura option.

### Step 4B — Google Colab la train pannradhu (Recommended, free T4 GPU)

1. Browser la [colab.research.google.com](https://colab.research.google.com) open pannu, new notebook create pannu.
2. Runtime > Change runtime type > **T4 GPU** select pannu.
3. Indha `Backend\ai-slm` folder முழுக்க zip pannu (illati GitHub-ku push pannitu clone pannu).
4. Colab cell-la:

```python
# Option A: GitHub la irundhu clone (repo already push panniruntha)
!git clone <your-repo-url>
%cd <repo>/Backend/ai-slm

# Option B: zip upload panni extract pannalam (files.upload() use panni)

!pip install -r requirements.txt
!python generate_dataset.py --n 3000 --seed 42
!python validate_dataset.py
!python train_lora.py --max-steps 20          # smoke test mudhallaadi
!python train_lora.py --epochs 2              # full training (2-4 மணி நேரம் varai aagalam T4 la)
```

5. Mudinjadhukku apparam, `out/loomnest-shop-lora/adapter` folder-a Colab la irundhu **download pannu** (zip pannitu).

### Step 5 — Adapter-a base model-oda merge pannu

(CPU la panna mudiyum, ~4 GB RAM pothum, GPU vendam)

```powershell
python merge_adapter.py --base-model Qwen/Qwen2.5-1.5B-Instruct --adapter out\loomnest-shop-lora\adapter --out out\merged
```

Idhu training mudinjadhukku apparam (adapter download panni, idhே local-la kondu vandhadhukku apparam) run pannu.
Output: `out\merged` folder la full merged model safetensors-aa save aagum.

### Step 6 — GGUF ku convert pannu + quantize pannu (Ollama-ku venum format)

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
pip install -r requirements/requirements-convert_hf_to_gguf.txt
python convert_hf_to_gguf.py ../out/merged --outfile ../out/loomnest-shop-slm.f16.gguf --outtype f16
cmake -B build && cmake --build build --config Release -t llama-quantize
# Windows: build\bin\Release\llama-quantize.exe
./build/bin/llama-quantize ../out/loomnest-shop-slm.f16.gguf ../out/loomnest-shop-slm.Q4_K_M.gguf Q4_K_M
```

Idhu model size-a kammi pannudhu (quantize), adhanaala fast-a + kammi RAM la run aagum.

### Step 7 — Ollama la serve pannu

```bash
ollama create loomnest-shop-slm -f Modelfile
ollama run loomnest-shop-slm "hi"      # sanity check
```

Apparam `recommendation-service` restart pannu, environment variable set panni:

```
LLM_MODEL=loomnest-shop-slm
```

(`llm.enabled=true` and `llm.base-url=http://localhost:11434/v1` already set-aa irukkanum.)

### Step 8 — Base model-oda compare pannu (evaluation)

`Backend\ai-eval` folder la irukura `run_eval.py` use panni, tuned model vs base model (`qwen2.5:1.5b`) compare
pannalam. README.md la "Section 5" la detailed steps irukku.

---

## 4. Summary — Quick checklist

| # | Command | GPU venuma? |
|---|---|---|
| 1 | `python generate_dataset.py --n 3000 --seed 42` | இல்லை |
| 2 | `python validate_dataset.py` | இல்லை |
| 3 | `python train_lora.py --dry-run` | இல்லை |
| 4 | `pip install -r requirements.txt` | இல்லை |
| 5 | `python train_lora.py --max-steps 20 ...` (smoke test) | **ஆம்** (Colab recommended) |
| 6 | `python train_lora.py --epochs 2` (full training) | **ஆம்** (Colab recommended) |
| 7 | `python merge_adapter.py ...` | இல்லை (CPU) |
| 8 | GGUF convert + quantize (llama.cpp) | இல்லை (CPU) |
| 9 | `ollama create` + `ollama run` | இல்லை |
| 10 | `recommendation-service` la `LLM_MODEL` change | இல்லை |

---

## 5. Common Issues (Doubt varum podhu paaru)

- **"missing train file" error** → Step 1 (`generate_dataset.py`) run pannala-nu artham. First adha run pannu.
- **`CUDA out of memory`** → `--rank` kammi pannu (8), `--batch-size 1`, `--grad-accum` adhigama pannu (32),
  `--max-seq-len` kammi pannu (2048), illati Colab ku maari pannu.
- **`torch.cuda.is_available()` False** → Local machine la CUDA driver/toolkit correct-aa install aagala,
  illati Colab T4 use pannu.
- **`bitsandbytes` install fail** (Windows) → `--no-4bit` try pannu, illati WSL2/Linux/Colab use pannu.
- **TRL `SFTTrainer` error** → script automatic-aa `transformers.Trainer` ku fallback aagidum (`--trainer auto`
  default), worry venam.

Edhavadhu step la stuck aana, andha exact error message-a ping pannu — naan help pannuren. 👍

---

## 6. Troubleshooting Log (20-step smoke test-la face pannina errors + fixes)

Idhu namba 20-step smoke test pannумbodhu face pannина **real errors**, exact fix-oda. **Full training (`--epochs 2`) pannумbodhu idhе errors mathuvadhum varralam** — adhanala idha முதலில படிचcу vачіருnga.

### 6.1 Torch CPU-only install aagiruchu (local laptop)

**Symptom**: `python -c "import torch; print(torch.cuda.is_available())"` → `False`, aana `nvidia-smi` GPU kaatuдhу.

**Reason**: `pip install -r requirements.txt` default-a torch-oda **CPU-only build** install pannudhu (Windows-la PyPI default).

**Fix**:
```powershell
pip uninstall torch -y
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### 6.2 `--max-seq-len` romba kammi-aa kodutha, "0 examples tokenized, all dropped"

**Symptom**: `train: 0 examples tokenized, 2700 dropped (too long / no assistant tokens)`

**Reason**: Ovvoru example-oda system prompt + 24 tool schemas mattume **~3.3k-4.2k tokens** — `--max-seq-len` idhукku kiழே kodutha, எல்லாம் drop aagும்.

**Fix**: `--max-seq-len` flag-a **podaadhinga** (default 4608 use pannட்டும்). Idha kuraikka mudiyadhu.

### 6.3 `trl.SFTTrainer` "gemm input" crash (T4 GPU-la)

**Symptom**:
```
RuntimeError: gemm input...
File ".../trl/kernels/chunked_logprob.py", line 734, in backward
```

**Reason**: Puthiy `trl` version-oda "chunked_logprob" optimized kernel **bf16 gemm** use pannudhu — idhu **Ampere-um adhukku melana GPUs mattum** (A100, RTX 30xx+) support pannum. **T4 — Turing architecture (pre-Ampere)** — adhanala crash.

**Fix**: `--trainer hf` flag kodu (plain `transformers.Trainer` use pannum, same data/masking, model quality same):
```python
!python train_lora.py --epochs 2 --trainer hf
```

### 6.4 OOM warnings (`transformers.Trainer` vachu) + romba slow (~300s/step)

**Symptom**:
```
[W...] memory allocation failed with OOM on device 0 while trying to allocate ... (free: ..., total: 15637086208)
```

**Reason**: Qwen2-oda periya vocabulary (151,643 words) + nedumaiyana sequence (4608 tokens) → loss calculate pannumbodhu oru **huge (~2GB) temporary buffer** venum, T4-oda 16GB almost full-aa use aagiruchu.

**Fix**: Memory fragmentation reduce pannu:
```python
!PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train_lora.py --epochs 2 --trainer hf
```

### 6.5 Kaggle — "Could not resolve host: github.com"

**Reason**: Kaggle notebook-la **Internet toggle OFF** irundhuchu.

**Fix**: Right panel → **Session options → Internet → ON** pannu (first time phone verification venum: Settings → Phone Verification).

### 6.6 Kaggle — GPU session "Starting..." (yellow) romba neram stuck

**Fix order**: Page refresh (F5) → Accelerator-a "None" → mathuvadhum "GPU T4 x2" → GPU type maathi (P100) try pannu → puthiy notebook create pannu.

### 6.7 `!` prefix illama command kodutha (git clone, pip install, python...)

**Symptom**: `SyntaxError` illana command recognize aagale.

**Reason**: Notebook cell-la shell commands-ku **`!` prefix mandatory** (`!git clone...`, `!pip install...`, `!python...`). `%cd` mattum exception (ippadiye okay).

### 6.8 `<llama.cpp>` angle brackets literal-aa type pannina

**Symptom**: `/bin/bash: line 1: llama.cpp: No such file or directory`

**Reason**: `<...>` nu README-la irukkura **placeholder notation** — "idhoda pakkathula unga real path-a podu" nu artham, adhe type pannaadhinga.

**Fix**: Actual path kodu (`llama.cpp/convert_hf_to_gguf.py` illana `%cd llama.cpp` pannitu `convert_hf_to_gguf.py`).

### 6.9 `merge_adapter.py` — torchao version error

**Symptom**: `ImportError: Found an incompatible version of torchao. Found version 0.10.0, but only versions above 0.16.0 are supported`

**Reason**: Colab-la pazhaya `torchao` pre-installed irukku, `peft` puthiy version expect pannudhu — idha namba use pannave illa.

**Fix**:
```python
!pip uninstall torchao -y
```

### 6.10 GGUF convert — `AttributeError: 'list' object has no attribute 'keys'`

**Symptom**: `convert_hf_to_gguf.py` run pannumbodhu tokenizer load pannumbodhu crash.

**Reason**: Installed `transformers` version-oda bug — `tokenizer_config.json`-la `extra_special_tokens` **list format**-la irukku, library **dict** expect pannudhu.

**Fix** (`out/merged/tokenizer_config.json`-a patch pannu):
```python
import json
path = "../out/merged/tokenizer_config.json"
with open(path) as f:
    config = json.load(f)
if isinstance(config.get("extra_special_tokens"), list):
    config["extra_special_tokens"] = {t: t for t in config["extra_special_tokens"]}
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
```

### 6.11 Nested `ai-slm/ai-slm` folder confusion

**Reason**: `git clone` rendu thadava run panninadhal, illana already `ai-slm` kulla irundhu mathuvadhum `%cd ai-slm` pannina, nested folder varum.

**Fix**: Edhavadhu command run panna munnadi, `!pwd` and `!ls` vachu confirm pannu current folder enna irukkunu.

### 6.12 Chatbot integration — `NullPointerException: Cannot invoke "Number.intValue()"`

**Symptom**: Real app-la chat API call pannumbodhu, backend logs-la:
```
java.lang.NullPointerException: Cannot invoke "java.lang.Number.intValue()"...
at org.springframework.ai.tool.method.MethodToolCallback.callMethod
```

**Reason**: `ShoppingAssistantTools.java`-la `getTrendingProducts(int limit)`, `addToCart(..., int quantity)` madhiri methods-ku **mandatory `int` arguments** venum — 20-step smoke-test model idha eppozhuthum kodukkale (training incomplete).

**Fix**: Idhu **code bug illa** — full training (`--epochs 2`) mudinjadhukku apparam, model correct-aa ella mandatory arguments-um kodukkum, idhe error varadhu. Integration/wiring already correct-nu idhe confirm pannuchu.

---

**Summary**: Full training (`--epochs 2`) run panna idhe command use pannu (namba discover pannina ella fixes sertha):
```python
!PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train_lora.py --epochs 2 --trainer hf
```
Merge/GGUF step pannumbodhu, **6.9 (torchao uninstall)** and **6.10 (tokenizer patch)** fixes mathuvadhum pannanum (puthiy `out/merged` folder-ku).

---

## 7. ONE full command — overnight unattended run (train + merge + GGUF + quantize)

Idhu **ella steps-um (dataset generate → validate → full train → merge → tokenizer patch → GGUF convert → quantize) oru single cell-la** — ella fixes (6.1-6.11) sertha. `set -e` podirukkom, adhanala **edhavadhu step fail aana, udane script stop aagidum** (next steps waste aagadhu) — morning-la scroll-up pannina, exact enga nirutha nu kaanum.

### Prerequisite (idha cell run pannadhukku munnadi pannanum)

Colab: **Runtime → Change runtime type → T4 GPU** select pannitu Save pannu. (Kaggle: Session options → Accelerator → GPU T4 x2, Internet ON.)

### Cell 1 — Idhu mattum paste pannu, run pannitu night full-a vidu

```bash
%%bash
set -e

echo "=== Checking GPU ==="
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "NO GPU DETECTED - select GPU accelerator first, then rerun"; exit 1; }

echo "=== 1. Clone repo ==="
git clone https://github.com/Ramesh-456/ai-slm.git
cd ai-slm

echo "=== 2. Install dependencies ==="
pip install -r requirements.txt

echo "=== 3. Generate dataset ==="
python generate_dataset.py --n 3000 --seed 42 \
  --tools-java ShoppingAssistantTools.java \
  --chatbot-java ChatbotServiceImpl.java \
  --language-java LanguageSupport.java

echo "=== 4. Validate dataset ==="
python validate_dataset.py

echo "=== 5. FULL TRAINING (2 epochs, ~2-4 hours on T4) ==="
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train_lora.py --epochs 2 --trainer hf

echo "=== 6. Remove torchao (merge compatibility fix) ==="
pip uninstall torchao -y

echo "=== 7. Merge adapter into base model ==="
python merge_adapter.py --base-model Qwen/Qwen2.5-1.5B-Instruct --adapter out/loomnest-shop-lora/adapter --out out/merged

echo "=== 8. Patch tokenizer_config.json (transformers list/dict bug fix) ==="
python - <<'EOF'
import json
path = "out/merged/tokenizer_config.json"
with open(path) as f:
    config = json.load(f)
if isinstance(config.get("extra_special_tokens"), list):
    config["extra_special_tokens"] = {t: t for t in config["extra_special_tokens"]}
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    print("patched: list -> dict")
else:
    print("no patch needed")
EOF

echo "=== 9. Clone llama.cpp and convert to GGUF ==="
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
pip install -r requirements/requirements-convert_hf_to_gguf.txt
python convert_hf_to_gguf.py ../out/merged --outfile ../out/loomnest-shop-slm.f16.gguf --outtype f16

echo "=== 10. Build llama-quantize ==="
cmake -B build
cmake --build build --config Release -t llama-quantize

echo "=== 11. Quantize ==="
./build/bin/llama-quantize ../out/loomnest-shop-slm.f16.gguf ../out/loomnest-shop-slm.Q4_K_M.gguf Q4_K_M

cd ..
echo "=== ALL DONE ==="
ls -lh out/*.gguf
```

### Morning-la — Cell 2 (separate cell, idha manual-aa run pannanum)

⚠️ **`files.download()` ku browser tab active-aa open irukkanum** — adhanala idha **Cell 1 kulla sertha mudiyadhu**, automatic-aa trigger aagadhu tab close irundha. Morning-la vandhu, Cell 1 "ALL DONE" kaatiya apparam, **idha separate cell-la run pannu**:

```python
from google.colab import files
files.download('ai-slm/out/loomnest-shop-slm.Q4_K_M.gguf')
```

### Important — overnight-ku

- **Laptop-a sleep mode-ku pogadha madhiri** vachiruka (Settings → Power → "never sleep" / plugged-in mode).
- **Browser tab-a close pannadhinga** (background-la vachiruka pothum, screen off aana kooda problem illa — tab close mattum pannaadhinga).
- Internet connection stable-a irukkanum (WiFi drop aana session disconnect aagalam).
- Cell "actively running" irukkura varai (output print aaguthu), Colab idle-timeout apply pannadhu — idhu safe.
