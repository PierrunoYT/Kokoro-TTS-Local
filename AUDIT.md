# Codebase Audit — Kokoro-TTS-Local

**Audited:** 2026-07-29 · **Audited commit:** `a423e0f` (master) · **Scope:** 10 Python modules
(~4,146 LOC), packaging, Docker, CI

Six parallel review agents swept the tree across security, core correctness, UI/CLI correctness,
concurrency/performance, architecture, and dependencies/testing/docs/CI.
**~170 raw findings, ~145 after deduplication.** Health score at audit time: **42/100**.

| Severity | Count |
|---|---|
| Critical | 4 |
| High | 21 |
| Medium | ~55 |
| Low | ~65 |

## Current status — last reconciled 2026-07-29 against `ff524ed`

The finding bodies below describe the code **as audited at `a423e0f`** and are kept unedited as the
historical record — their line numbers and file paths predate the `src/` layout move. Each finding
carries a status marker reflecting the current tree:

- **🟢 RESOLVED** — re-verified fixed against the current source.
- **🔴 OPEN** — re-verified still present.
- **🟡 PARTIAL** — some sub-claims fixed, others still present; the marker says which.
- **⚪ OBSOLETE** — the code the finding described no longer exists (e.g. `config.py` was deleted).
- **◻️ NOT RE-VERIFIED** — not re-checked in this reconciliation pass. Status unknown, not "fine".

| Severity | Resolved | Open / Partial | Obsolete | Not re-verified |
|---|---|---|---|---|
| Critical (4 detailed) | 4 | 0 | 0 | 0 |
| High (15 detailed) | 14 | 1 (DOC-01) | 0 | 0 |
| Medium (21 detailed) | 4 | 16 | 1 | 0 |
| Low (30 tabulated) | 5 | 17 | 0 | 8 |

**The critical and high tier is closed except DOC-01**, a three-line README correction that is the
oldest untouched item in the report — the README still says "8 languages" in three places while
`VOICE_PREFIX_TO_LANGUAGE_CODE` maps nine. The medium tier is largely untouched: the remaining
sixteen are concentrated in `chinese_config.py` (config/text handling), the Gradio UX error paths,
and dependency pinning — none of which any remediation commit has reached yet.

Fixes landed in three commits: `2be12c6` (critical), `d87f7ed` (high), `ff524ed` (regressions found
by reviewing the first two). See the remediation log below.

### How to read this document

- **✅ VERIFIED** — at audit time, confirmed directly against source, by parsing the file, or by
  running the failing command. 13 findings. Treat as fact.
- **⚠️ REPORTED** — agent-reported and spot-checked but not individually re-derived. Strong leads,
  not settled facts. Confirm before acting.

**Known limits.** The `kokoro` package is not installed in this tree and is version-unpinned, so any
claim depending on `KPipeline` internals (including whether its internal `torch.load` sets
`weights_only=True`) is inferred from in-repo evidence, not observed. Reachability of a few Gradio
input-validation findings depends on the installed Gradio major version, likewise unpinned.

## Remediation log

### 2026-07-29 — Critical findings resolved

- **CORE-01:** `build_model()` now constructs `KModel` from the resolved checkpoint and matching
  repository config and passes that concrete model to `KPipeline`. English and Chinese configs are
  stored separately. Cache reuse includes checkpoint, config, repository revision, and device, so
  explicit fine-tunes are not discarded on a language-only cache hit.
- **CORE-02:** Chinese voices download into a temporary directory and are moved into the flat
  `voices/` directory that setup verification and the demos consume.
- **DEPLOY-01:** Compose now provides a valid `environment` mapping and supports `HF_HUB_OFFLINE`.
- **DATA-01:** Speed-dial mutations are protected by a process lock and use fsynced temporary files plus
  atomic replacement. Corrupt JSON now aborts mutations instead of being treated as an empty preset set.

### 2026-07-29 — High findings resolved

- **CONC-01 / FMT-01:** Web outputs use UUID-qualified names, AAC uses FFmpeg's ADTS muxer, and
  converted intermediates are removed.
- **CORE-03 / ARCH-01:** Voice routing uses one strict prefix mapping and rejects malformed or unknown
  prefixes instead of silently selecting English or inferring Chinese from a parent directory name.
- **LEGAL-01:** Wheel metadata now declares Apache-2.0 and includes `LICENSE`.
- **SEC-01:** Non-loopback Gradio binding requires credentials; Compose requires both credentials,
  publishes only on host loopback, and persists all application data in one managed volume.
- **SEC-02:** The Claude workflow is restricted to trusted repository associations, uses immutable
  action SHAs, removes OIDC write permission, and limits duplicate runs with concurrency controls.
- **DEAD-01 / ARCH-02:** The download lock protects atomic artifact promotion, inert generation
  parameters and duplicate mappings were removed, every component honors the shared path helpers, and
  the unused `config.py` abstraction was deleted.
- **CONC-02 / CONC-03 / SHUTDOWN-01:** Pipelines are cached by complete immutable identity, compatible
  language pipelines share one model-family lock, construction is single-flight outside registry
  locks, inference locks span lazy iteration, and idempotent shutdown drains work before releasing
  models.
- **PKG-01:** Runtime modules now install only inside the `kokoro_tts_local` namespace, console entry
  points use that package, and mutable state defaults to a stable platform user-data directory.
- **TEST-01:** Dependency-free regression tests cover checkpoint/config selection, persistence,
  language routing, concurrent construction/inference, and shutdown; a least-privilege CI workflow
  runs tests, compilation, wheel construction, installation, and entry-point checks.

### 2026-07-29 — Review of the remediation commits

Findings from reviewing `2be12c6` and `d87f7ed` themselves. Six issues, all introduced or left
standing by those two commits; all resolved here.

- **REG-01 (Windows console):** The regression suite failed on Windows — `UnicodeEncodeError` on the
  first Chinese `print` under a legacy cp1252 code page — and the same crash reached users through
  the newly added `kokoro-tts-setup` entry point. `console.enable_utf8_console()` now reconfigures
  the streams to UTF-8 with replacement, but only when the current encoding cannot represent the
  output. CI gained a `windows-latest` leg plus a run pinned to `PYTHONIOENCODING=cp1252` that
  reproduces the original failure.
- **REG-02 (download serialization):** DEAD-01 widened `_download_lock` to span `hf_hub_download`
  itself, so the voice-download thread pool ran strictly one file at a time — 54 serialized fetches
  on first run. The lock now covers only the existence re-check and the atomic promotion; the fetch
  into each worker's private temporary directory is unlocked and parallel again.
- **REG-03 (generator lock lifetime):** CONC-03 made `iter_speech` hold the model-family lock for the
  generator's whole lifetime, but consumers abandon that generator on `break` (the Gradio
  `max_segments` cap, the CLI timeout guards). The lock was then held through file writing and
  format conversion, and could block `shutdown_pipelines`, which drains by acquiring every family
  lock. All consumers now use `contextlib.closing`; a regression test asserts the lock is
  reacquirable immediately after `close()`.
- **REG-04 (unroutable voice names):** CORE-03 turned an unknown voice prefix from a silent English
  default into a `ValueError`, but the raise was unguarded on paths fed by directory enumeration, so
  a stray `.pt` file turned a dropdown selection into an unhandled traceback. `list_available_voices`
  now filters names the router would reject and logs each one.
- **REG-05 (explicit checkpoint):** `build_model` downloaded the repository default to whatever path
  the caller supplied, disguising a mistyped `model_path` as a working fine-tune. An explicit path is
  now required to exist. The CLI passed the managed default path explicitly and would have lost
  first-run download, so it now passes `None` and lets `build_model` own path resolution.
- **REG-06 (stale references):** Dead `generate_speech` imports across the three CLIs, its docstring
  still documenting the removed `lang`/`device` parameters, and a Chinese guide instruction pointing
  at the deleted `initialize_phonemizer`.

---

## Executive summary

*Written at audit time against `a423e0f`. Retained as the original assessment; see the status block
above for what has since been fixed.*

The security posture at the application layer is better than expected. The correctness posture is worse.

**The single most consequential finding:** `build_model()` resolves, downloads and validates a model
checkpoint — then never passes it to the pipeline. Every user who downloads the 330 MB Chinese
checkpoint is running English v1.0 weights while being told *"中文模型加载成功 (Chinese model loaded
successfully)"*.

A theme runs through the whole codebase: **work is performed and then silently discarded.** The Chinese
setup script downloads all 8 voices into `voices/voices/` and then verifies `voices/`. `config.py` is a
243-line "centralized configuration system" with zero consumers. `_download_lock` is declared and never
acquired. `generate_speech()` accepts a `lang` parameter it never reads. AAC export calls an ffmpeg
muxer that does not exist.

Where the code is good, it is genuinely good: `get_safe_voice_path()` is a correct allow-list validator
with resolve-then-contain ordering, used consistently at every user-facing call site; offline mode is
threaded correctly through every download; the container runs as non-root; there is no shell execution
or unsafe deserialization in the repo's own code. The problem is not carelessness — it is the absence
of any test or CI that would have caught the silent failures.

### Postscript — 2026-07-29, after remediation

Every one of the "silently discarded work" instances above is now fixed, and the missing net exists:
five dependency-free regression tests run on Linux and Windows in CI.

The remediation itself proved the thesis. Reviewing the two fix commits turned up six new defects
**introduced by the fixes** (REG-01…REG-06) — a suite that could not run on Windows at all, a lock
widened until it serialized a thread pool, an inference lock that outlived its generator, and a
`ValueError` promoted into a code path that enumerates a user-writable directory. Three of the six
trace to a fix that was correct in isolation and wrong in context.

What remains open is qualitatively different from what was closed: no remaining finding silently
produces wrong output. They are unpinned dependencies, un-surfaced UI errors, and text-handling bugs
in the Chinese config module, which no remediation commit has touched.

---

## Top 10 by impact

Status as of `ff524ed`. Locations are as-audited (`a423e0f`), before the `src/` layout move.

| # | Issue | Location | Sev | Effort | Status |
|---|---|---|---|---|---|
| 1 | Downloaded checkpoint never reaches the pipeline | `models.py:610` | Critical | M | 🟢 CORE-01 |
| 2 | Chinese setup downloads to `voices/voices/` | `setup_chinese_tts.py:201` | Critical | S | 🟢 CORE-02 |
| 3 | `docker compose` fails to start | `docker-compose.yml:9` | Critical | S | 🟢 DEPLOY-01 |
| 4 | Speed-dial preset data loss | `speed_dial.py:107` | Critical | S | 🟢 DATA-01 |
| 5 | Concurrent users overwrite each other's audio | `gradio_interface.py:217` | High | S | 🟢 CONC-01 |
| 6 | Any path containing "zh" switches to the Chinese model | `models.py:506` | High | S | 🟢 CORE-03 |
| 7 | Docker publishes an unauthenticated UI on `0.0.0.0` | `Dockerfile:43` | High | S | 🟢 SEC-01 |
| 8 | CI: unpinned `@beta` action, any-user trigger, `id-token: write` | `claude.yml:15-33` | High | S | 🟢 SEC-02 |
| 9 | AAC export always fails | `gradio_interface.py:147` | High | S | 🟢 FMT-01 |
| 10 | Package declares MIT; repo is Apache-2.0 | `pyproject.toml:10` | High | S | 🟢 LEGAL-01 |

---

# Critical

## CORE-01 🟢 RESOLVED · ✅ VERIFIED — The downloaded model checkpoint is never used

**Severity:** Critical · **Files:** `models.py:477-655`, construction at `:610`

`build_model()` spends ~40 lines resolving `model_path`, selecting the `hexgrad/Kokoro-82M-v1.1-zh`
repo for Chinese, and downloading the checkpoint. **The final reference to `model_path` is the log line
at `:544`.** The pipeline is then built as:

```python
pipeline_instance = EnhancedKPipeline(lang_code=lang_code)   # line 610 — no model_path
```

And `EnhancedKPipeline.__init__` (`:161`) only forwards `lang_code` and `model=True` to `KPipeline`,
which loads its own default checkpoint.

**Failure scenario.** `chinese_tts_demo.py:248` passes `kokoro-v1_1-zh.pth`. The file downloads
(~330 MB), is validated by `chinese_config.validate_chinese_model()`, then discarded. Synthesis runs the
English v1.0 checkpoint with Chinese voice packs and a `z` G2P front-end — degraded Mandarin, reported
as success. Any user-supplied fine-tune is ignored identically. The weights are also effectively
downloaded twice (once to `model_dir`, once into the HF cache by `KPipeline`).

**Fix.** Plumb the checkpoint through, and share one `KModel` across language pipelines.

```python
class EnhancedKPipeline(KPipeline):
    def __init__(self, lang_code='a', model=True, repo_id=None):
        kwargs = {'repo_id': repo_id} if repo_id else {}
        super().__init__(lang_code=lang_code, model=model, **kwargs)

# build_model(), replacing line 610
repo_id = "hexgrad/Kokoro-82M-v1.1-zh" if is_chinese_model else "hexgrad/Kokoro-82M"
pipeline_instance = EnhancedKPipeline(lang_code=lang_code, repo_id=repo_id)
```

---

## CORE-02 🟢 RESOLVED · ✅ VERIFIED — Chinese setup downloads every voice into `voices/voices/`

**Severity:** Critical · **Files:** `setup_chinese_tts.py:201-206`, verify at `:250`, `VOICES_DIR` at `:28`

```python
downloaded_path = hf_hub_download(
    repo_id="hexgrad/Kokoro-82M",
    filename=f"voices/{voice_file}",     # repo-relative subdir
    local_dir=str(VOICES_DIR),           # already ".../voices"
)
```

`hf_hub_download` preserves the repo subdirectory under `local_dir`, so files land in
`voices/voices/zf_xiaobei.pt`. `verify_setup()` then checks `VOICES_DIR / voice_file`.

**Failure scenario.** On a clean checkout all 8 downloads print `✓ 完成 (Done)`, then all 8 verification
checks print `✗`, the summary says "Setup Incomplete", and the script exits 1. `list_available_voices()`
globs `voices/*.pt` non-recursively and finds nothing, so `chinese_tts_demo.py` aborts with "No Chinese
voices found". Re-running re-downloads everything, because the `voice_path.exists()` short-circuit at
`:189` never fires.

Note `models.py:398-417` gets this right — temp dir, then `shutil.move`.

**Fix.** Reuse `models.download_voice_files()`, or mirror its temp-dir-then-move pattern:

```python
import tempfile, shutil
with tempfile.TemporaryDirectory() as tmp:
    downloaded_path = hf_hub_download(
        repo_id="hexgrad/Kokoro-82M",
        filename=f"voices/{voice_file}",
        local_dir=tmp,
    )
    shutil.move(downloaded_path, str(voice_path))
```

---

## DEPLOY-01 🟢 RESOLVED · ✅ VERIFIED — Docker Compose refuses to start

**Severity:** Critical · **File:** `docker-compose.yml:9-11`

`environment:` is followed only by comment lines, so it parses as `null` rather than a mapping.

```
$ docker compose config
validating docker-compose.yml: services.kokoro-tts.environment must be a mapping
```

The documented Docker path is non-functional as shipped.

**Fix.** Give the key a real mapping (this is also where the DEPLOY-02 credentials belong):

```yaml
    environment:
      HF_HUB_OFFLINE: "0"
      KOKORO_TTS_USERNAME: ${KOKORO_TTS_USERNAME:?}
      KOKORO_TTS_PASSWORD: ${KOKORO_TTS_PASSWORD:?}
```

---

## DATA-01 🟢 RESOLVED · ✅ VERIFIED — Speed-dial presets can be silently wiped

**Severity:** Critical · **Files:** `speed_dial.py:99-112` (save), `:124-141` (delete), `:53-55` (load)

Three compounding defects:

1. `open(SPEED_DIAL_FILE, 'w')` truncates to zero bytes **before** `json.dump` runs.
2. `load_presets` swallows `json.JSONDecodeError` and returns `{}` (`:53-55`).
3. Read-modify-write with no lock — Gradio serves concurrent sessions.

**Failure scenario.** User has 20 presets. A save is interrupted (Ctrl+C, disk full, or a second
concurrent save) leaving the file truncated. The *next* `save_preset` calls `load_presets`, gets `{}`
from the parse failure, and writes a file containing only the new preset. **All 20 originals are
permanently gone**, with one console line as the only trace. No backup exists.

**Fix.** Module lock + write-temp-then-`os.replace` (atomic on Windows and POSIX). Never treat a parse
failure as "empty" on a write path — rename the bad file to `.corrupt` instead.

```python
import os, tempfile, threading
_presets_lock = threading.RLock()

def _atomic_write(data: dict) -> bool:
    fd, tmp = tempfile.mkstemp(dir=str(SPEED_DIAL_FILE.parent or "."),
                               prefix=".speed_dial.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, SPEED_DIAL_FILE)
        return True
    except OSError as e:
        print(f"Error writing speed dial presets: {e}")
        try: os.unlink(tmp)
        except OSError: pass
        return False
```

The same non-atomic pattern exists in `config.py:171-179` and `chinese_config.py:225-245` — one helper
fixes all three.

---

# High

## CONC-01 🟢 RESOLVED · ✅ VERIFIED — Concurrent users overwrite (and receive) each other's audio

**Severity:** High · **File:** `gradio_interface.py:217-219`

```python
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")   # one-second resolution
base_name = f"tts_{timestamp}"
```

No PID, UUID or session component; `sf.write` overwrites unconditionally.

**Failure scenario.** Two users click Generate in the same wall-clock second — routine for short text on
GPU. Both resolve to `tts_20260729_141530.wav`. Depending on interleaving, user A downloads a truncated
file **or receives user B's audio**. Given this UI is explicitly built for network sharing
(`--host 0.0.0.0`, "Network sharing capabilities" in the module docstring), that is a cross-user content
leak. `convert_audio` amplifies it: `AudioSegment.from_wav` reads the WAV *after* generation, so it can
read a file another thread already overwrote.

**Fix.**

```python
import uuid
base_name = f"tts_{timestamp}_{uuid.uuid4().hex[:8]}"
```

Also unlink the intermediate WAV after mp3/aac conversion (`:283-284`) — it currently leaks.

---

## CORE-03 🟢 RESOLVED · ✅ VERIFIED — Any path containing "zh" switches to the Chinese model

**Severity:** High · **File:** `models.py:506`

```python
is_chinese_model = lang_code == 'z' or (model_path and 'zh' in str(model_path).lower())
```

The substring test runs against the **whole absolute path**, and `tts_demo.py:48` passes a fully-resolved
`Path`.

**Failure scenario.** A user at `C:\Users\zhang\kokoro\` or `D:\zhongwen\Kokoro-TTS-Local\` runs the
English CLI. `is_chinese_model` becomes true → the Chinese checkpoint is downloaded from
`hexgrad/Kokoro-82M-v1.1-zh` → `initialize_phonemizer('zh')` at `:579` clobbers the English phonemizer →
`af_bella` gets Chinese G2P. Nothing in the output explains why.

Secondary: the expression returns `None`, not `False`, when `model_path is None`.

**Fix.**

```python
is_chinese_model = bool(
    lang_code == 'z'
    or (model_path is not None and 'zh' in Path(str(model_path)).name.lower())
)
```

---

## FMT-01 🟢 RESOLVED · ✅ VERIFIED — AAC export has never worked

**Severity:** High · **File:** `gradio_interface.py:147`

`audio.export(path, format="aac")` passes `aac` to ffmpeg as `-f`. There is no `aac` output muxer — the
raw-AAC muxer is named `adts`. Confirmed empirically:

```
$ ffmpeg -i t.wav -f aac -y out.aac
Error initializing the muxer for out.aac: Invalid argument

$ ffmpeg -i t.wav -f adts -c:a aac -y out2.aac
size=2KiB  time=00:00:00.20  bitrate=85.1kbits/s      # works
```

**Failure scenario.** The user picks "aac" and clicks Generate. `CouldntEncodeError` is swallowed by the
broad handler at `:161`, `convert_audio` returns `None`, and the audio player goes blank **with no error
in the browser** — the traceback only reaches the server console. An orphaned WAV is left in `outputs/`.

**Fix.**

```python
elif format.lower() == "aac":
    audio.export(str(output_path), format="adts", codec="aac", bitrate="192k")
```

---

## LEGAL-01 🟢 RESOLVED · ✅ VERIFIED — Package metadata declares the wrong license

**Severity:** High · **Files:** `pyproject.toml:10` vs `LICENSE:1-2`, `README.md:591`

`LICENSE` is the full Apache License 2.0 and the README states Apache 2.0 twice, but `pyproject.toml`
declares `license = { text = "MIT" }`. There is no MIT text anywhere in the repo.

> Introduced by the packaging commit `5948cd0` in this session — my error. Worth correcting first, since
> any wheel published with it carries the wrong terms.

**Why it matters.** SBOM tooling and compliance gates record MIT. Apache-2.0 carries an express patent
grant and NOTICE/attribution obligations that consumers relying on the declared MIT terms would skip.

**Fix.**

```toml
license = "Apache-2.0"
license-files = ["LICENSE"]
classifiers = ["License :: OSI Approved :: Apache Software License"]
```

---

## SEC-01 🟢 RESOLVED · ✅ VERIFIED — Docker ships an unauthenticated UI bound to all interfaces

**Severity:** High · **Files:** `Dockerfile:43`, `docker-compose.yml:7-8`, `gradio_interface.py:613`

The CLI default is a safe `127.0.0.1`, but the container CMD overrides it to `--host 0.0.0.0` and passes
no credentials, so `auth=None`. Compose publishes `7860:7860`, binding all host interfaces and punching
through the host firewall via Docker's NAT rules on Linux. The auth flags appear nowhere in the README
(no hits for `--username`, `--password`, `KOKORO_TTS_USERNAME`).

**Why it matters.** Anyone who can reach port 7860 gets free GPU compute, an unthrottled DoS vector (no
queue depth or rate limit, no compose resource limits), and read access to generated audio — which, per
CONC-01, may be another user's. `./outputs` and `./voices` are bind-mounted to the host.

**Fix.** Publish to loopback only, and refuse to bind a non-loopback host without credentials:

```python
LOOPBACK = {"127.0.0.1", "::1", "localhost"}
if auth is None and args.host not in LOOPBACK:
    sys.exit(f"Refusing to bind {args.host} without authentication. "
             "Set KOKORO_TTS_USERNAME/PASSWORD or bind 127.0.0.1.")
```

```yaml
    ports:
      - "127.0.0.1:7860:7860"
```

---

## SEC-02 🟢 RESOLVED · ✅ VERIFIED — CI: mutable action ref, any-user trigger, OIDC write

**Severity:** High · **File:** `.github/workflows/claude.yml:15-19`, `:25`, `:33`

Three issues compound:

1. `anthropics/claude-code-action@beta` is a **mutable ref** whose code runs with `ANTHROPIC_API_KEY`.
2. The trigger fires for **any GitHub account** — no `author_association` check; the only gate is the
   literal string `@claude`.
3. `id-token: write` (`:25`) is granted on every such run.

**Why it matters.** A stranger commenting `@claude hi` burns your Anthropic quota on demand, repeatable
in a loop. If `beta` is ever force-pushed or upstream is compromised, the next drive-by comment
exfiltrates the key — the `experimental_allowed_domains` egress restriction is commented out (`:36-44`).
`id-token: write` lets the job mint an OIDC token that over-broad cloud trust policies would accept.

**Not a pwn-request:** there is no `pull_request_target`, `actions/checkout@v4` uses `fetch-depth: 1`
with no `ref:`, so untrusted PR head code is never checked out. Token scopes (`contents/pull-requests/
issues: read`) are correctly least-privilege.

**Fix.** Pin the action to a full SHA, gate on `author_association`, drop `id-token`, add a concurrency
group:

```yaml
    if: |
      contains(fromJSON('["OWNER","MEMBER","COLLABORATOR"]'),
               github.event.comment.author_association || github.event.issue.author_association)
      && ( ... existing @claude checks ... )
    concurrency:
      group: claude-${{ github.event.issue.number }}
      cancel-in-progress: true
    permissions:
      contents: read
      pull-requests: read
      issues: read
      # id-token removed
```

---

## DEAD-01 🟢 RESOLVED · ✅ VERIFIED — Three declared-but-inert mechanisms

**Severity:** High · **Files:** `models.py:321`, `models.py:751`, `chinese_config.py:156`

**`_download_lock`** — declared with the comment "Lock for download operations", acquired nowhere in the
repo (verified by grep). It reads as protection over exactly the racy download paths (non-atomic
`shutil.move` into the live voices dir) and provides none.

**`generate_speech(lang=…)`** — the parameter appears only in the signature (`:751`) and docstring
(`:761`), never in the body. Calling `generate_speech(model, "你好", voice="zf_xiaoxiao", lang='z')`
against an `'a'` pipeline yields English G2P over Mandarin — the exact bug class the
`fix/gradio-language-routing` branch was created to fix, still live in the public API.

**`KOKORO_VOICES_DIR`** — honoured by `models.get_voices_dir()`, ignored by `chinese_config.py:156`
(`Path("voices").resolve()`), `config.py:52` and `setup_chinese_tts.py:28`. Set it as `README.md:200`
documents and `list_available_voices()` finds 54 voices while `get_chinese_voices()` returns `[]` in the
same process. `ensure_voices_directory()` additionally creates a stray empty `<cwd>/voices`.

---

## DOC-01 🔴 OPEN · ✅ VERIFIED — "8 languages" is wrong; there are 9

**Severity:** Medium · **Files:** `README.md:8`, `:322`, `:480`

`VOICE_PREFIX_TO_LANGUAGE_CODE` (`models.py:244-254`) contains 9 distinct codes: `a, b, e, f, h, i, j,
p, z`. `LANGUAGE_CODES` (`models.py:232-242`) likewise has 9 entries. The README says "8 languages" in
three places.

---

## ARCH-01 🟢 RESOLVED · ⚠️ REPORTED — Three competing prefix→language mappings

**Severity:** High · **Files:** `models.py:244-254`, `gradio_interface.py:69-79`, `chinese_config.py:29`

`gradio_interface.LANG_MAP` is a duplicate of `models.VOICE_PREFIX_TO_LANGUAGE_CODE` that has already
drifted: `gradio_interface.py:75` defines `"fm_": "f"`, a prefix that exists nowhere else in the repo
(verified). Two sources of truth for this mapping is how the language-routing bug fixed on this branch
arose in the first place.

Additionally `get_language_code_from_voice` (`models.py:705-715`) matches a bare 2-char prefix with no
`_` boundary check, and falls back silently to `'a'`:

- `important.pt` → prefix `im` → **Italian** pipeline
- `emma_custom.pt` → `em` → **Spanish** G2P over English text
- Future `kf_`/`vf_` (Korean/Vietnamese) voices → silently English

**Fix.** Delete `LANG_MAP`; require the `_` boundary; warn on unknown prefixes instead of defaulting.

---

## ARCH-02 🟢 RESOLVED · ⚠️ REPORTED — `config.py` is a 243-line dead abstraction

**Severity:** High · **File:** `config.py` (entire), `chinese_config.py:154-158`

`config.py` describes itself as a "Centralized Configuration System" and has **zero consumers**. The
only import of it (`chinese_tts_demo.py:43`) never uses the symbol. Meanwhile:

| Setting | `config.py` | Actual source of truth |
|---|---|---|
| `voice_files` | `:85-121` | `models.py:193-229` — **byte-identical, 36 lines** |
| `language_codes` | `:72-82` | `models.py:232-242`, plus a third variant in `gradio_interface.py:69-79` |
| `sample_rate` | `:29` | 15 literal `24000` occurrences across 6 files |
| `min/max_speed` | `:32-33` | 16 occurrences across 5 files, **two conflicting ranges** (0.1–3.0 vs 0.5–2.0) |
| `voices_dir` | `:52` | `models.py:35-44`, `chinese_config.py:156`, `setup_chinese_tts.py:28` |

`ChineseTTSConfig` is separately a ~91% copy-paste of `TTSConfig`. `TTSConfig.validate_speed`,
`validate_language`, `save_config` and `set_config` have zero call sites; `validate_sample_rate` is
reimplemented byte-for-byte in `gradio_interface.py:52` and `tts_demo.py:27`.

**Fix.** Either delete `config.py` (−244 LOC) or adopt it fully. Do not leave it as-is.

---

## CONC-02 🟢 RESOLVED · ⚠️ REPORTED — Global lock held across hundreds of MB of network I/O

**Severity:** High · **File:** `models.py:497-655`

`with _pipeline_lock:` opens at `:497` and does not close until `:655`. Inside it: the model download
(~330 MB, `:536-543`), the config download (`:562-569`), `initialize_phonemizer` (`:576-587`), and
`download_voice_files` (`:591`) which itself downloads up to 54 voice files.

**Failure scenario.** First user on a cold install takes the lock for 2–10 minutes. Every other thread
touching `build_model`, `load_voice` (`:739`) or `generate_speech` (`:781`) blocks — including threads
that only wanted an already-loaded English pipeline. The server appears completely hung to all users;
there is no timeout, so a stalled HF connection wedges the process indefinitely.

**Related:** the single-slot `_pipeline` cache (`:319`, `:500-502`, `:649`) is keyed **only** on
`lang_code`, so `model_path`, `repo_version` and `device` are silently dropped on a cache hit, and every
language switch rebuilds from scratch while leaking the evicted pipeline (no `del`, no
`empty_cache()`). `models.load_voice()` calls `build_model` on *every* invocation, so alternating
between `af_bella` and `zf_xiaobei` triggers a full rebuild each time.

---

## CONC-03 🟢 RESOLVED · ⚠️ REPORTED — Generation runs outside the lock on a false premise

**Severity:** High · **File:** `models.py:781-817`, comment at `:795-797`

The comment asserts the generator "only reads from the cache". `EnhancedKPipeline.load_voice`
(`:167-190`) is an **override** that writes `self.voices[voice_name] = voice_model.to(self.device)`
(`:184`), and `KPipeline.__call__` resolves its `voice=` argument by calling `self.load_voice(...)` —
which by Python dispatch is that override. So the generator performs an unsynchronized `dict.__setitem__`
plus a `torch.load` plus a `.to(device)` on the process-wide shared pipeline, entirely outside the lock.

Compounding: `model.device = device` is written under the lock at `:783` but read at `:184` during the
unlocked generation.

---

## SHUTDOWN-01 🟢 RESOLVED · ⚠️ REPORTED — Cleanup destroys the live model mid-generation

**Severity:** High · **File:** `gradio_interface.py:453-572`, `:609-621`

`cleanup_resources` is registered via `atexit` (`:551`) and installed as the SIGINT/SIGTERM handler
(`:557-572`). Python delivers signals on the **main** thread; Gradio serves requests on worker threads.
The handler clears `model.voices`, iterates `dir(model)` and sets every attribute with a `.to` method to
`None`, then calls `torch.cuda.empty_cache()` and `sys.exit(0)` — none of it synchronized against
workers.

**Failure scenario.** Ctrl+C while a generation is streaming → the worker dereferences a now-`None`
attribute (`AttributeError`), or `empty_cache()` releases allocator blocks while a kernel is still
executing (CUDA illegal-memory-access). The routine also runs **twice** on normal exit (`finally` +
`atexit`) and **three times** on Ctrl+C, and never touches the `pipelines` dict (`:80`) at all — so the
per-language pipelines it is supposed to free are leaked, and the "freed X MB" report prints ~0.

---

## PKG-01 🟢 RESOLVED · ⚠️ REPORTED — Flat layout installs 9 generic top-level modules

**Severity:** High · **File:** `pyproject.toml:56-66`

`py-modules` installs `models`, `config`, `tts_demo`, `speed_dial`, … as top-level names in
site-packages. `import config` and `import models` are maximally collision-prone with other packages in
the same environment.

Related: a non-editable `pip install` produces console scripts that misbehave, because every default path
(`speed_dial.py:19`, `gradio_interface.py:62`, `tts_demo.py:49`, `chinese_tts_demo.py:47`) is relative to
the **current working directory**, not the package. Running `kokoro-tts` from a different directory
silently creates a different `speed_dial.json` / `outputs/` / `output.wav`.

**Fix.** Move the modules under a `kokoro_tts_local/` package and resolve user data against
`platformdirs`-style locations (or at minimum `get_base_dir()`), not `Path.cwd()`.

---

## TEST-01 🟢 RESOLVED · ⚠️ REPORTED — No test suite exists

**Severity:** High · **File:** `test_offline.py`

`test_offline.py` is an interactive smoke script, not a test suite: it cannot run in CI, requires network
access to download real model files, and is excluded from the installed package while `README.md` tells
installed users to run it.

Zero coverage of:

- **Language routing** — the thing this branch exists to fix, and the source of ARCH-01/CORE-03
- **`get_safe_voice_path`** — a security-relevant path-traversal guard
- **The `KOKORO_*` path helpers** — where DEAD-01 and CFG-01 live
- **Speed-dial persistence** — where DATA-01 lives
- **Audio conversion** — where FMT-01 lives

Every one of this audit's verified findings would have been caught by a test in one of those five areas.

---

# Medium

## CFG-01 🟢 RESOLVED · ⚠️ REPORTED — `KOKORO_CONFIG_PATH` honoured for the check, ignored for the download

**File:** `models.py:551-573`

```python
config_path = str(get_config_path())          # e.g. /etc/kokoro/config.json
if not os.path.exists(config_path):
    config_path = hf_hub_download(..., local_dir=str(model_dir))   # writes <model_dir>/config.json
```

The download lands in `model_dir`, not at `KOKORO_CONFIG_PATH`, so the configured path never appears.
**Every startup re-runs the HF request**, and hard-fails the moment `HF_HUB_OFFLINE=1` is set — even
though the config was "downloaded" on the previous 20 runs.

## CFG-02 🔴 OPEN · ⚠️ REPORTED — `ChineseTTSConfig` ignores its own `paths.voices_dir`

**File:** `chinese_config.py:154-158`

`self.chinese_voices_dir` is set **before** the config file loads and never reassigned. A
`chinese_tts_config.json` setting `paths.voices_dir` is parsed, merged, exposed via `get()`, and has no
effect whatsoever.

## CFG-03 ⚪ OBSOLETE · ⚠️ REPORTED — `validate_sample_rate` can return an invalid rate

**File:** `config.py:188-201`

The "default" is read from the very config the user may have corrupted:
`{"audio": {"sample_rate": 8000}}` → logs "Using default rate: 8000" → **returns 8000**. Kokoro always
emits 24 kHz, so the WAV plays back 3× too slow. `validate_language` (`:209`) has the same
self-referential pattern.

> **Status:** `config.py` was deleted in `d87f7ed` (ARCH-02), so this exact code is gone. The
> replacement `validate_sample_rate`/`validate_language` in `gradio_interface.py` and `tts_demo.py`
> use hard-coded defaults rather than reading them back from user config, so the pattern did not
> survive the move.

## CFG-04 🟡 PARTIAL · ⚠️ REPORTED — `_merge_config` accepts arbitrary types

**Files:** `config.py:135-144`, `chinese_config.py:201-210`

No type or schema check. `{"language_codes": "abc"}` makes `validate_language` execute `"abc".keys()` →
uncaught `AttributeError`. `{"paths": "voices"}` produces a misleading "key not found" error.

> **Status:** the `config.py` half is gone with the file. `chinese_config._merge_config` is unchanged
> and still merges arbitrary types with no schema check.

## TEXT-01 🔴 OPEN · ⚠️ REPORTED — `normalize_chinese_text` destroys the newlines the splitter needs

**File:** `chinese_config.py:106-119`, consumed at `chinese_tts_demo.py:283`

`' '.join(text.split())` collapses **all** whitespace including `\n`. Every synthesis path splits on
`split_pattern=r'\n+'`. A 5-paragraph article therefore becomes one chunk, hits Kokoro's 510-token limit,
and is **silently truncated** — a 2,000-character input yields ~15 s of audio covering the first ~400
characters, with no error. `chinese_tts_demo.py:284` logs only `text[:50]`, hiding it.

## TEXT-02 🔴 OPEN · ⚠️ REPORTED — `is_chinese()` misclassifies Japanese

**File:** `chinese_config.py:97-103`

Only checks `U+4E00–9FFF`. `is_chinese_text("日本語です")` → `True` (kanji are in that block), so
auto-routing sends Japanese to `zf_*` voices → Mandarin readings of Japanese kanji. Conversely misses
Ext-A/Ext-B and fullwidth forms, warning "not Chinese" on legitimately Chinese input.

## CHI-01 🔴 OPEN · ⚠️ REPORTED — Chinese demo discards good audio on `None` phonemes

**File:** `chinese_tts_demo.py:331`

`" ".join(all_phonemes)` raises `TypeError` if any segment yielded `ps=None`. Caught by the broad handler
at `:337` → returns `(None, None)` → the user waits through a full synthesis, sees every segment log
success, then gets "生成失败" and **all audio is discarded**. `gradio_interface.py:252` and
`models.py:813` both guard with `if ps:`; this file is the odd one out.

## CHI-02 🟢 RESOLVED · ⚠️ REPORTED — Setup fetches the v1.0 config for the v1.1-zh model

**File:** `setup_chinese_tts.py:156-160`

Model comes from `hexgrad/Kokoro-82M-v1.1-zh`; `config.json` comes from the v1.0 repo. If the configs
diverge (vocab, istftnet params) the pipeline is configured for the wrong checkpoint.

## UX-01 🔴 OPEN · ⚠️ REPORTED — Every Gradio failure returns `None`; the user sees nothing

**File:** `gradio_interface.py:288-292`

Empty text, missing voice file, ffmpeg absent, CUDA OOM, corrupted `.pt` — all produce an identical
blank audio player. A remote user (this is a network-shared UI) cannot distinguish "still running" from
"ffmpeg missing". The 5,000-char truncation at `:212-214` is likewise console-only.

**Fix.** Return `(audio, status)` and render the status in a Textbox.

## UX-02 🔴 OPEN · ⚠️ REPORTED — Preset handlers push plain strings into a Dropdown value

**File:** `gradio_interface.py:388-390`, `:403-405`

Error paths return `gr.update(value="Please provide a name, voice, and text")` where the output is a
`gr.Dropdown` whose `choices` are preset names. The dropdown ends up displaying the error message as its
selected value, and a subsequent "Load" passes that string to `get_preset()`.

`load_preset_fn` (`:377-385`) separately returns `None` for a `gr.Slider`, which is not a valid value —
the next Generate sends `speed=None` into the pipeline.

## CLI-01 🔴 OPEN · ⚠️ REPORTED — `tts_demo.py` deletes the previous output before validating the new audio

**File:** `tts_demo.py:126-139`

`output_path.unlink()` runs unconditionally, *then* `if audio_data is None or len(audio_data) == 0: raise`.
A failed generation leaves the user with **no** `output.wav` at all. `ValueError` is not in the
`(IOError, PermissionError)` handler, so it pointlessly retries a deterministic failure 3× with 2 s
sleeps.

## CLI-02 🔴 OPEN · ⚠️ REPORTED — Per-segment timeout silently truncates long generations

**File:** `tts_demo.py:299-347`, `MIN_GENERATION_TIME = 60` at `:18`

`segment_start_time` is anchored *before* the generator is created, so the first interval includes
pipeline warm-up and voice loading. On CPU a 9,000-character input (well under the 10,000 limit enforced
at `:261`) exceeds 60 s on the first chunk → `break` → partial audio is saved and announced as
`Audio saved to …`. The 300 s overall cap does the same.

## PERF-01 🔴 OPEN · ⚠️ REPORTED — No `torch.inference_mode()` anywhere

**Files:** `models.py:799-814`, `gradio_interface.py:233-253`, `tts_demo.py:321-358`,
`chinese_tts_demo.py:308-322` (verified absent repo-wide by grep)

Every generation builds a full autograd graph, retaining intermediate activations. For a 5,000-character
generation split into ~100 segments this is substantial wasted memory per concurrent request — a direct
path to CUDA OOM under exactly the multi-user load this app targets.

## PERF-02 🔴 OPEN · ⚠️ REPORTED — `dependency_checker` imports everything and may spawn 13 subprocesses

**File:** `dependency_checker.py:58-83`

`importlib.import_module` for all 13 packages including `torch`, `gradio` and `spacy`, then falls back to
`subprocess.run([... 'pip', 'show', pkg], timeout=10)` per package. Worst case **130 s of blocking
startup**. No memoization — `check_dependencies()` redoes everything on every call. Should use
`importlib.metadata.version` instead.

## DEP-01 🔴 OPEN · ⚠️ REPORTED — 27 of 28 dependencies fully unpinned

**Files:** `requirements.txt`, `pyproject.toml:16-48`

Only `numpy<2.0` is constrained. Every `docker compose build` produces a different image from the same
source, so a supply-chain compromise is neither reproducible nor detectable by diffing. This also makes
the Gradio-version-dependent findings unpredictable.

Also: `maturin` (a build backend), `wheel` and `setuptools` are declared as **runtime** dependencies;
`underthesea` (Vietnamese NLP) is a hard dependency for an unsupported language.

**Fix.** `pip-compile --generate-hashes`, commit the lock, install with `--require-hashes`.

## SEC-03 🔴 OPEN · ⚠️ REPORTED — Model artifacts fetched from a mutable revision with no integrity check

**File:** `models.py:323`, `:402-413`, `:536-543`

Everything is pulled from `hexgrad/Kokoro-82M@main`, rewritable at any time by the upstream owner. The
only "integrity verification" is `st_size == 0` — and `hashlib` is imported at `:339` and **never used**,
while the comment at `:411` says "Verify file integrity". Because `.dockerignore` excludes `voices/` and
`*.pth`, every container build downloads fresh.

## SEC-04 🟢 RESOLVED · ⚠️ REPORTED — Unvalidated `format` reaches `mkdir(parents=True)`

**File:** `gradio_interface.py:282-284`, `:139`

`format` is interpolated into a `Path` and `.resolve()`d, then `output_path.parent.mkdir(parents=True)`
runs **before** the allow-list check at `:145-150`. A `format` of `"../../../../x"` creates directories
outside `outputs/`. Bounded (directory creation, not file write) and reachability depends on the Gradio
version's `Radio.preprocess` validation — but it is defence you do not control.

## SEC-05 🔴 OPEN · ⚠️ REPORTED — `speed` is not validated server-side

**File:** `gradio_interface.py:167`, `:233`

`MIN_SPEED`/`MAX_SPEED` exist at `:44-45` and are never referenced; `config.validate_speed` is never
called. The slider's `minimum`/`maximum` are client-side only. `speed=1e-9` asks the vocoder to stretch a
5,000-character utterance by a factor of a billion — unauthenticated OOM in one request. `speed=0` and
`float('nan')` also pass through (`nan` defeats both range comparisons).

## SEC-06 🟢 RESOLVED · ⚠️ REPORTED — Global `json.load` monkey-patch races other threads

**File:** `models.py:124-138`, applied at `:609`

The context manager swaps the **process-global** `json.load` while `EnhancedKPipeline()` is constructed
(a multi-second, network-bound window). The docstring claims `_pipeline_lock` makes this safe — it only
serializes `build_model` callers, not Gradio request threads, `huggingface_hub`, or `spacy`.
`safe_json_load` also unconditionally `fp.seek(0)` and reads `fp.buffer`, bypassing the `TextIOWrapper`'s
decoding.

**Fix.** Strip the BOM from the file once on disk; never patch a global.

## ERR-01 🔴 OPEN · ⚠️ REPORTED — Error handling is inconsistent and lossy

**Counts:** 8 bare `except:`, 57 broad `except Exception`, 13 `traceback.print_exc()`

`models.generate_speech` collapses empty input, invalid voice, missing file, OOM and device mismatch into
an undifferentiated `(None, None)` (`:816-827`), and the caught tuple includes `TypeError`/`AttributeError`
— so genuine programming errors (e.g. a `kokoro` signature change) surface as "no audio". The 8 bare
`except:` clauses swallow `KeyboardInterrupt` and `SystemExit`, which during the shutdown paths makes the
process feel unkillable.

## DOC-02 🟡 PARTIAL · ⚠️ REPORTED — Further documentation drift

- `dependency_checker.py` is documented as checking memory, disk space and audio — it checks none.
- The documented Chinese setup flow is broken end-to-end (CORE-02).
- Gradio auth flags (`--username`/`--password`) and the `KOKORO_*` env vars are undocumented.
- Chinese voice "quality grades" in `CHINESE_TTS_GUIDE.md` contradict the main README and appear fabricated.
- The README configuration example uses a key that does not exist.
- Project-structure listing omits every packaging/Docker/CI file.
- Documented speed ranges are wrong in both the CLI and web sections.

> **Status:** the README rewrite in `d87f7ed` documented the auth flags and `KOKORO_*` env vars, fixed
> the project-structure listing, and the Chinese setup flow now works end-to-end (CORE-02). The
> `dependency_checker` description, the disputed voice "quality grades", the non-existent config key,
> and the speed ranges were not revisited. The stale `initialize_phonemizer` reference introduced by
> the same rewrite was corrected in `ff524ed`.

---

# Low

Grouped; each verified only as a count or by grep unless noted.

| ID | Finding | Location | Status |
|---|---|---|---|
| L-01 | `list(model.__dict__.keys())` missing → `RuntimeError: dictionary changed size during iteration` on **every** clean exit | `tts_demo.py:441` | 🟢 |
| L-02 | Watchdog timer not cancelled on exception paths — fires 5 min later during an unrelated prompt | `tts_demo.py:309-379` | 🔴 |
| L-03 | `Ctrl+C` in `tts_demo.py` dumps a raw traceback (`KeyboardInterrupt` is not `Exception`); `chinese_tts_demo.py:460` handles it correctly | `tts_demo.py:419` | 🔴 |
| L-04 | Default voice fallbacks (`af_bella`, `zf_xiaobei`) are not validated against installed voices | `tts_demo.py:75`, `chinese_tts_demo.py:167` | ◻️ |
| L-05 | Legacy voice-migration comparison is always true (`Path('voices') != Path('C:/…/voices')`); "moved" files are actually `copy2`-ed and never deleted | `models.py:677-700` | 🔴 |
| L-06 | Phoneme/audio segment lists desynchronize when `ps` is empty | `models.py:806-817` | 🔴 |
| L-07 | `initialize_phonemizer` probes non-English languages with Chinese text (`'测试'`), and leaves stale globals on failure. Both globals are write-only — never read anywhere | `models.py:258-310` | 🟢 |
| L-08 | espeak-ng's Mandarin code is `cmn`, not `zh` — the Chinese branch fails on most installs and is swallowed | `models.py:579` | 🟢 |
| L-09 | `EnhancedKPipeline.load_voice` narrows its parent's contract (path-only, drops `delimiter`, keys by `stem`) — breaks blended voices and will break on a `kokoro` upgrade | `models.py:167-190` | 🔴 |
| L-10 | Import-time side effects: `logging.basicConfig` ×3 hijacking the root logger, `signal.signal`, `atexit`, an espeak probe, two config singletons doing disk I/O | `models.py:17,313`, `gradio_interface.py:550-572` | 🟡 |
| L-11 | `os.environ["PYTHONIOENCODING"] = "utf-8"` at import is a no-op — the interpreter reads it only at startup | `models.py:146` | 🔴 |
| L-12 | `OFFLINE_MODE` frozen at import; setting `HF_HUB_OFFLINE` later has no effect | `models.py:151` | 🔴 |
| L-13 | Empty/whitespace env vars: `KOKORO_BASE_DIR="   "` creates a directory literally named three spaces | `models.py:31-68` | 🔴 |
| L-14 | 30 unused imports and dead locals (pyflakes-verified); `chinese_tts_demo.py:43` imports `TTSConfig` and never uses it | all modules | 🟡 |
| L-15 | `download_voice_files` bypasses `get_safe_voice_path` — latent arbitrary-overwrite if a future caller plumbs user input in | `models.py:352,417` | 🔴 |
| L-16 | `download_voice_files([])` raises "check your internet connection"; `required_count` is bypassed on the early-return path | `models.py:352,367-370` | 🔴 |
| L-17 | Predictable temp filename + unlink/rename TOCTOU (symlink-following write when CWD is shared) | `tts_demo.py:146-158` | ◻️ |
| L-18 | Full user text logged at INFO on every request → Docker json-file log with no rotation | `gradio_interface.py:222,251` | ◻️ |
| L-19 | `outputs/` grows without bound; intermediate WAV never deleted after conversion | `gradio_interface.py:62,277` | 🟡 |
| L-20 | `split_chinese_text` is O(n²) via `+=` string concatenation; degenerates to one segment per character when `max_length <= 0` | `chinese_config.py:132-148` | ◻️ |
| L-21 | Preset names reject non-ASCII (a project shipping full Chinese TTS); `"demo"` and `"demo "` become distinct keys | `speed_dial.py:83,103` | 🔴 |
| L-22 | `validate_chinese_model()` returns `True` for a 2 KB truncated download — the size check only warns | `chinese_config.py:260-271` | ◻️ |
| L-23 | `tts_demo.main()` is 298 lines at 9 levels of nesting; `chinese_tts_demo.py` duplicates the entire CLI scaffolding | `tts_demo.py:227-417` | ◻️ |
| L-24 | 319 `print()` vs 100 `logger.*`; 4 files use only `print` | all modules | ◻️ |
| L-25 | Container runs as UID 10001 against host-owned bind mounts — writes fail on Linux | `Dockerfile:31-35` | 🟢 |
| L-26 | `.gradio/certificate.pem` committed (it is the public ISRG Root X1 CA — not a secret, but noise) | `.gradio/` | 🔴 |
| L-27 | Unpinned base image, no build cache mount, no OCI labels, no healthcheck | `Dockerfile` | 🟡 |
| L-28 | Per-request imports inside hot functions (`import psutil` on every Generate click) | `gradio_interface.py:179` | ◻️ |
| L-29 | Unreachable `else` branch and dead `return` | `gradio_interface.py:231-235`, `models.py:433` | 🟢 |
| L-30 | `models.generate_speech` and `models.load_voice` — 110 LOC of public API with **zero callers**; all three front-ends hand-roll the generator loop instead | `models.py:717-827` | 🔴 |

---

# Quick wins — all verified, under one hour total

**9 of 10 done.** Only #3 remains, and it is a 3-minute edit.

| # | Change | Time | Status |
|---|---|---|---|
| 1 | Correct the package license to Apache-2.0 in `pyproject.toml` | 2 min | 🟢 |
| 2 | Make `environment:` a mapping so Docker Compose starts at all | 2 min | 🟢 |
| 3 | Fix the language count — "8 languages" → 9, in three README locations | 3 min | 🔴 |
| 4 | Switch AAC to the `adts` muxer with explicit `codec="aac"` | 5 min | 🟢 |
| 5 | Match "zh" against the filename, not the whole absolute path | 5 min | 🟢 |
| 6 | Add a UUID suffix to generated output filenames | 5 min | 🟢 |
| 7 | Delete `_download_lock` or wire it into the download path | 5 min | 🟢 |
| 8 | `list(model.__dict__.keys())` in `tts_demo.py:441` — stops the `RuntimeError` on every clean exit | 2 min | 🟢 |
| 9 | Delete `LANG_MAP` (and its phantom `fm_`); import the mapping from `models.py` | 10 min | 🟢 |
| 10 | Honour or remove `generate_speech(lang=…)` | 15 min | 🟢 |

---

# Remediation plan

Effort: **S** <1h · **M** ~half a day · **L** ~2–3 days · **XL** a week or more

Status as of `ff524ed`. Phases 1, 3 and 8 are essentially complete; 2 has not been started.

| # | Phase | Contains | Effort | Status |
|---|---|---|---|---|
| 1 | **Stop the bleeding** | All ten quick wins. Restores Docker, AAC and license correctness; removes the two silent mis-routings. | S | 🟡 9/10 — quick win #3 outstanding |
| 2 | **Make failures visible** | Surface errors in the Gradio UI instead of returning `None` (UX-01); replace the 8 bare `except:`; stop `generate_speech` collapsing every failure into `(None, None)`. **Do this before the deeper fixes so you can see them work.** | M | 🔴 Not started — **now the highest-value remaining phase** |
| 3 | **Fix the model path** | CORE-01 and CORE-02 — plumb the checkpoint into `KPipeline`; rewrite Chinese setup to reuse `models.download_voice_files()`. | M | 🟢 Done |
| 4 | **Durable state** | DATA-01 atomic writes + lock, applied to `speed_dial.py` and both config classes (identical bug, one helper). | S | 🟡 `speed_dial.py` done; `chinese_config.py` still writes non-atomically |
| 5 | **Harden the deployment** | SEC-01 loopback publish + mandatory auth off-loopback; SEC-02 pin the action, gate on author association, drop `id-token`; commit a hash-pinned lock file (DEP-01). | M | 🟡 SEC-01/SEC-02 done; DEP-01 lock file outstanding |
| 6 | **Add the missing net** | A real `pytest` suite plus a CI workflow. Start with the five areas from TEST-01 — each would have caught a verified finding above. Add `ruff` for the 30 dead imports. | L | 🟡 `unittest` suite + two-OS CI exist; no `ruff`, dead imports remain |
| 7 | **Collapse the duplication** | Delete or fully adopt `config.py`; make `ChineseTTSConfig` subclass `TTSConfig`; one `get_voices_dir()` everywhere; extract shared CLI scaffolding. Roughly −500 LOC. | L | 🟡 `config.py` deleted and paths unified via `paths.py`; CLI scaffolding still duplicated |
| 8 | **Concurrency model** | Per-language pipeline cache with eviction; network I/O outside the global lock; per-pipeline lock held across generation; drain-aware idempotent shutdown; `torch.inference_mode()` at all four generation sites. | XL | 🟡 Done except `torch.inference_mode()` (PERF-01) and cache eviction |

### One structural note

Most concurrency findings share a single root cause: **one mutable pipeline object is shared across all
users while its identity — language, device, voice cache — is reassigned per request.** Making the
pipeline immutable after construction and caching per language collapses roughly a dozen separate
findings at once. That is the highest-leverage refactor in phase 8 and is worth doing before the smaller
concurrency patches.

**Outcome (2026-07-29).** This was done: pipelines are keyed by full immutable identity, `device` and
`lang_code` raise on reassignment after publication, and one `KModel` plus one lock is shared per
model family. It collapsed CONC-02, CONC-03 and SHUTDOWN-01 as predicted. It also introduced REG-03 —
holding the family lock across lazy iteration is correct, but nothing made the generator's owners
close it. The refactor was right; its blast radius was one step larger than the plan accounted for.

### Suggested next steps

1. Quick win #3 (3 min) — closes the last High.
2. Phase 2, "make failures visible" — still untouched and still the prerequisite for trusting
   anything else. UX-01 alone means a remote user cannot tell a missing ffmpeg from a slow request.
3. PERF-01 `torch.inference_mode()` — one decorator at each generation site, and the only remaining
   item from the otherwise-complete concurrency phase.
4. Delete the dead `os.environ["PYTHONIOENCODING"] = "utf-8"` at `models.py:85` (L-11). It has never
   done anything — the interpreter reads that variable only at startup — and now sits next to
   `console.py`, which solves the problem it was reaching for.

---

*Generated by a six-agent parallel audit. Findings marked ✅ VERIFIED were confirmed directly against
source or by execution; ⚠️ REPORTED findings are agent-reported leads that should be confirmed before
acting. Status markers (🟢🔴🟡⚪◻️) were added in the 2026-07-29 reconciliation pass and reflect the
tree at `ff524ed`, not the audited commit.*
