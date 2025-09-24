yep — the globs were the issue (case + exact filenames). here are **fixed, copy-pasteable** commands for the four that didn’t work, plus equivalent `llama.cpp` direct-download lines if you want to skip HF CLI.

---

### Llama 3.2 1B Instruct (use **bartowski** repo; filenames are `Llama-3.2-1B-Instruct-*.gguf`)

```bash
huggingface-cli download bartowski/Llama-3.2-1B-Instruct-GGUF --local-dir llama3_1b \
  --include "Llama-3.2-1B-Instruct-Q2_K.gguf" \
            "Llama-3.2-1B-Instruct-Q3_K_M.gguf" \
            "Llama-3.2-1B-Instruct-Q4_K_M.gguf" \
            "Llama-3.2-1B-Instruct-Q5_K_M.gguf"
```

Or via `llama.cpp`:

```bash
./build/bin/llama-cli -m hf://bartowski/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q4_K_M.gguf -p "hi" -n 8
```

(Exact filenames listed on the repo page.) ([Hugging Face][1])

---

### Qwen 2.5-0.5B Instruct (official **Qwen** repo; **lowercase** filenames like `qwen2.5-0.5b-instruct-q*_*.gguf`)

```bash
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct-GGUF --local-dir qwen05b \
  --include "qwen2.5-0.5b-instruct-q2_k.gguf" \
            "qwen2.5-0.5b-instruct-q3_k_m.gguf" \
            "qwen2.5-0.5b-instruct-q4_k_m.gguf" \
            "qwen2.5-0.5b-instruct-q5_k_m.gguf"
```

Or via `llama.cpp`:

```bash
./build/bin/llama-cli -m hf://Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf -p "hi" -n 8
```

(File list shows these exact names; case matters.) ([Hugging Face][2])

---

### Gemma 2 2B Instruct (use **bartowski**; this repo has `IQ3_M`, not `IQ2_M`)

```bash
huggingface-cli download bartowski/gemma-2-2b-it-GGUF --local-dir gemma2_2b \
  --include "gemma-2-2b-it-IQ3_M.gguf" \
            "gemma-2-2b-it-Q3_K_L.gguf" \
            "gemma-2-2b-it-Q4_K_M.gguf"
```

Or via `llama.cpp`:

```bash
./build/bin/llama-cli -m hf://bartowski/gemma-2-2b-it-GGUF/gemma-2-2b-it-Q4_K_M.gguf -p "hi" -n 8
```

(Exact filenames shown on the model page.) ([Hugging Face][3])

---

### Phi-3 Mini 4K Instruct

* The **Microsoft** repo only has a single `…-q4.gguf`. For **IQ2 / IQ3 / Q3\_K\_M / Q4\_K\_M** variants, use **bartowski**:

```bash
huggingface-cli download bartowski/Phi-3-mini-4k-instruct-GGUF --local-dir phi3mini \
  --include "Phi-3-mini-4k-instruct-IQ2_M.gguf" \
            "Phi-3-mini-4k-instruct-IQ3_M.gguf" \
            "Phi-3-mini-4k-instruct-Q3_K_M.gguf" \
            "Phi-3-mini-4k-instruct-Q4_K_M.gguf"
```

Or via `llama.cpp`:

```bash
./build/bin/llama-cli -m hf://bartowski/Phi-3-mini-4k-instruct-GGUF/Phi-3-mini-4k-instruct-IQ3_M.gguf -p "hi" -n 8
```

(These filenames are listed verbatim on the page.) ([Hugging Face][4])

---

#### Quick tips

* **Globs are case-sensitive.** Don’t mix `Q4_K_M` with `q4_k_m`.
* If an include returns “0 files,” open the repo’s **Files** tab and copy the **exact** filename.
* `llama.cpp` can **auto-download** from HF with `hf://repo/path/File.gguf` if you prefer not to use HF CLI. ([Hugging Face][5])

Want me to fold these into your existing bash loop so it pulls only what’s missing before benchmarking?

[1]: https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF "bartowski/Llama-3.2-1B-Instruct-GGUF · Hugging Face"
[2]: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/tree/main "Qwen/Qwen2.5-0.5B-Instruct-GGUF at main"
[3]: https://huggingface.co/bartowski/gemma-2-2b-it-GGUF "bartowski/gemma-2-2b-it-GGUF · Hugging Face"
[4]: https://huggingface.co/bartowski/Phi-3-mini-4k-instruct-GGUF "bartowski/Phi-3-mini-4k-instruct-GGUF · Hugging Face"
[5]: https://huggingface.co/docs/hub/en/gguf-llamacpp?utm_source=chatgpt.com "GGUF usage with llama.cpp"
