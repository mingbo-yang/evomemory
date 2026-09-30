# glm-4-9b load hang — layer isolation report

**Scope:** CPU-only. No GPU was touched (no CUDA context created; `CUDA_VISIBLE_DEVICES=""` for the model-load test).
Read-only on all data; scratch only under `/tmp`. No process was killed.
Python: `/home/ymb/miniconda3/envs/qwen35/bin/python` (3.10.19, safetensors 0.7.0, torch present).

## Headline

**The blocking layer is the NFS server's read path for one single file —
`model-00002-of-00004.safetensors` — not safetensors, not mmap, and not the file's
structure/content.** NFS READ requests for that one file never complete; because the
mount is `hard,timeo=600,retrans=2`, the kernel retries forever and never returns an
error, so readers sit in uninterruptible `D` state with `rchar` frozen — exactly the
observed "blocked, not I/O-starved" signature.

Two premise corrections are important (details in Steps 3 and 4):
1. `/mnt/huawei/ymb` and `/mnt/huawei/wwq` are **the same NFS export**, not different ones.
2. Every "safetensors load succeeds" test is **zero-copy** and never reads data, so it
   cannot detect this fault. The tests that actually touch bytes are the ones that hang.

---

## Step 1 — Does safetensors know the file? (lazy metadata probe)

`safe_open(shard2, framework="pt")` → `.keys()` → `.get_slice()` → `.metadata()`.

**Completed: 1.857 s total** (open 1.857 s, keys/get_slice/metadata ~0.000 s).
- 132 keys, e.g. `model.layers.10.input_layernorm.weight`, `...mlp.gate_up_proj.weight`
- slice shape `[4096]`, metadata `{'format': 'pt'}`

Control, identical probe on `Qwen3-8B/model-00001-of-00005.safetensors`: **~1.9–2.2 s**
(5 repeats 1.90–2.23 s; Qwen 3 repeats 2.14–2.21 s). **No difference between the two.**

This is lazy/zero-copy, so it only proves the header is parseable — it does not touch tensor data.

Also validated every shard's header/offset/size arithmetic statically (no data read):

| shard | size on disk | header len | `8+hdr+data_end` | status | keys |
|---|---|---|---|---|---|
| glm-00001 | 4 984 133 600 | 12 248 | 4 984 133 600 | exact | 107 |
| glm-00002 | 4 895 075 168 | 15 192 | 4 895 075 168 | exact | 132 |
| glm-00003 | 4 895 075 184 | 15 208 | 4 895 075 184 | exact | 132 |
| glm-00004 | 4 025 669 744 | 8 296 | 4 025 669 744 | exact | 72 |

`sum(sizes) - sum(8+hdr) = 18 799 902 720` = `index.json metadata.total_size` **exactly**.
Key sets match the index weight_map exactly (no missing/unexpected keys). Qwen's 5 shards
likewise exact. **No truncation, no index/header mismatch.**

## Step 2 — Force a REAL read through the safetensors layer (shard 2 only, ~4.9 GB)

| variant | result | time |
|---|---|---|
| `load_file(path)` (default mmap) | "completed", 132 tensors / 4.90 GB | 3.06 s |
| `load_file(path, device="cpu")` | "completed", 132 tensors / 4.90 GB | **0.031 s**, RSS **+7 MB** |
| `open(path,'rb').read()` → `safetensors.torch.load(buf)` | **HUNG** | `timeout 180` fired |

**Critical methodological finding:** the first two are *not* evidence of readability.
`device="cpu"` grew RSS by only 7 MB for 4.90 GB of logical tensors, i.e. safetensors 0.7.0
returns **mmap-backed, still-lazy** tensors even with `device="cpu"`. A follow-up
`load_file` over **all four** glm shards "loaded" 18.80 GB in **0.03 s** — physically
impossible as a copy, confirming nothing was read.

The only Step-2 variant that truly materialises bytes — the plain byte-buffer read —
**hard-stalled and `timeout 180` had to kill it.** Its watchdog showed
`rchar = 2164.0 MB` frozen and `RSS = 3145.9 MB` frozen for every sample from +5 s to
+175 s: it never advanced by a single byte in three minutes.

## Step 3 — Pure mmap on the NFS file vs local disk, independent of safetensors

- `mmap.mmap(fd, 0, ACCESS_READ)` itself: **instant (0.0001 s)** — mapping is free.
- Touch of *cached* pages: instant (0.0000 s at 0 MB, 1000 MB, 2769 MB).
- A 300 s full-mmap-scan variant **HUNG** (exit 124); its output was lost to Python
  stdout buffering (my error — re-run with `-u`).
- One bounded 4 KB-page scan of shard 2 from 2700 MB → EOF (1 MB stride, 1969 faults)
  **DID complete: 51.11 s (~38 MB/s effective)**. So the tail is not 100 % dead when
  touched as small, readahead-assisted page faults — an honest nuance that makes
  "entirely unreadable" too strong.
- **Local XFS baseline (`/dev/sdc3`, `/tmp`)**: `mmap()` 0.00002 s, touches 0.00001 s,
  full 200 MB scan **0.040 s = 5.3 GB/s**. Local disk is not involved in the fault.

**Premise correction:** `/mnt/huawei`, `/mnt/huawei2` and `/mnt/huawei/wwq` all report
**`dev=50`, fstype `nfs`, and `172.25.76.194:/cxc`** — one export, mounted twice. There is
no "different NFS export" separating glm-4-9b from the working models.

## Step 4 — Control: a model that does load (`Qwen3-8B`, same export)

All O_DIRECT (cache-bypassing) 4 MB reads, same client, same moments:

| target | @ offset 0 | @ 2000 MB |
|---|---|---|
| glm-00001 | DONE 2 s | DONE 2 s |
| glm-00002 | **BLOCKED >20 s, state D** | **BLOCKED** (also 2600/3000/4600/4620/4640/4660 MB) |
| glm-00003 | DONE 2 s | DONE 2 s |
| glm-00004 | DONE 2 s | DONE 2 s |
| Qwen3-8B s1 | DONE 2 s | DONE 2 s |
| Qwen3-8B s2 | DONE 2 s | — |

- Qwen full 4 GB O_DIRECT read: **progressed steadily at 22.6 MB/s** (2.69 GB in 119 s),
  never wedged.
- A `gemma3` model under the *same* `/mnt/huawei/ymb` subtree: O_DIRECT 10 MB @1000 MB in 1.36 s.
- Page-cache residency (`mincore`): glm shards 1/3/4 and Qwen shard1 = **100.0 %**;
  **glm shard 2 = 59.7 %, contiguous resident prefix = 2 769 682 432 bytes.**

**So the fault is specific to one file, not to the export, the client, the mount, or the model family.**

Independent corroboration (read-only inspection of another agent's logs, `/home/ymb/glm_local/logs/`):
- `copy_model-00001/00003/00004` all `DONE` (≈5–7 s each) — shards 1/3/4 read fine.
- `copy_model-00002` never finishes (local copy truncated at 2 769 289 216 bytes).
- `verify.log`: sha256 `OK` for shards 1, 3, 4 and `tokenizer.json`; **shard 2 is the only
  MISMATCH**, purely because its local copy is short.

**Every blocked reader stalls at the same place — the end of the cached prefix:**
`sha256sum` rchar = 2 769 691 251; `cp` rchar = 2 769 308 709; `open().read()` at 2164 MB;
`mincore` prefix = 2 769 682 432. That is also exactly why loaders report
"Loading safetensors checkpoint shards: **25% | 1/4**": shard 1 loads (100 % cached),
then shard 2 wedges forever.

`/proc/<pid>/syscall` shows the blocked processes parked in read (`syscall_nr=0`).

## Step 5 — Model-dir metadata

- `ls -la`: 4 shards 4.98/4.90/4.90/4.03 GB, owner `isaac:yl`, mode 777 (readable by us).
- `.cache/huggingface/download/` exists, holds `.metadata` + `.lock` for every file and
  **no `.incomplete` files** — the download completed cleanly.
- `manifest_sha256.json` values are **byte-identical to the HF LFS etags** stored in the
  `.metadata` files — consistent, and independently confirmed by the parent's own sha256
  run (shards 1/3/4 OK).
- Sizes vs `index.json`: exact to the byte (table in Step 1). No shard is truncated; no
  unexpected mtime.
- **Anomaly worth flagging:** all five glm files share an identical `ctime` of
  **2026-09-09 03:30:43** while their `mtime`s are April — a bulk metadata operation on
  2026-09-09. Qwen's files by contrast have `ctime == mtime` (2025-11-04). Circumstantial
  only (the tree also shows migration tooling), but consistent with this tree having been
  migrated/relocated recently — a plausible way for one file's data to end up unservable.
- `dmesg`: the server does log `nfs: server 172.25.76.194 not responding, still trying`
  and `__nfs4_reclaim_open_state: Lock reclaim failed!` — **but the newest such entry is
  2026-08-07, ~36 days ago.** I am *not* claiming those events caused the current stall;
  they only show this server has a history of going unresponsive.

## Mechanism (why it spreads, and why `rchar` freezes)

Late evidence: the first 200 MB of shard 2 are still **100 % mincore-resident**, yet
`dd bs=1M count=200` over that region now blocks in `D` state having read **zero file
bytes** (its `rchar` is 5329, which is just the dynamic loader's own reads — `ld.so.cache`
etc.; the progress log is empty). The identical dd on a local file does 200 MB at 2.8 GB/s
in 0.07 s. Pages that are cached but **locked by wedged in-flight NFS I/O** block every
new reader, so the file becomes progressively more unreadable as requests pile up —
**25 processes** currently hold shard 2 open in `D` state. At the start of this
investigation the same 200 MB read took 0.07 s.

Causal chain: server won't serve shard 2 → `hard` mount retries forever, never returns EIO
→ reader blocked in `D` with frozen `rchar` → page-cache pages stay locked → even cached
regions become unreadable → any consumer needing the whole file (transformers, vLLM,
sha256sum, cp, dd) hangs at 1/4 progress.

## Verdict

**Which layer blocks: the NFS read path (server side) for the single file
`model-00002-of-00004.safetensors`.** Not safetensors-mmap as a mechanism, not the NFS
export/client, not the file's structure or content.

**Single most decisive piece of evidence:** at the same moment, on the same mount, from the
same client, O_DIRECT 4 MB reads of glm shards 1/3/4 and both Qwen shards complete in
~2 s, while the same read of glm shard 2 blocks in uninterruptible `D` state at *every*
offset tried (0, 1000, 2600, 3000, 4600, 4620, 4640, 4660 MB) — with `mincore` showing
shard 2 only 59.7 % cached as a 2 769 682 432-byte prefix, exactly where every independent
reader (`sha256sum`, `cp`, plain `read()`) froze.

**Explicitly still inconclusive:**
- The exact server-side cause (lost/unavailable blocks, HSM/tiering recall, or transient
  overload) cannot be determined from the client. The uniform 2026-09-09 `ctime` is a lead,
  not proof.
- I could not hash shard 2 end-to-end, so I cannot state whether its readable 56.6 % matches
  upstream — but that is moot: the file is unreadable, which fully explains the hang.
- One 1 MB-stride mmap page scan of the tail *did* finish (51 s), so the tail is not 100 %
  dead for small readahead-assisted faults; read failure is request-dependent. This does not
  change the verdict, since every synchronous read that misses cache wedges.

**Recommended next actions (not taken — outside my read-only scope):**
1. Treat shard 2 as corrupt/unavailable: re-obtain `model-00002-of-00004.safetensors` and
   verify against sha256 `029329f34bb73bd71b7e5f6391c62383d6bf888b88d4168fb7c416ea283da3f7`.
2. Do **not** re-run loaders against the current file — each attempt adds another wedged
   process and further locks page-cache pages.
3. ~25 processes are wedged in `D` state on this file (mine included). They cannot be killed
   until their I/O completes; they are read-only and harmless, but they will need reaping
   after the NFS layer recovers.
4. Raise this with whoever runs `172.25.76.194:/cxc` — and consider `soft`/`intr`-style
   timeouts, since `hard` is what converts a server-side stall into a permanent silent hang.
