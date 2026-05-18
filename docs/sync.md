# Sync — Local &harr; GCP Server

This document describes how to (a) connect to the GCP servers that run the LSF pipeline, and (b) keep the local checkout and the server checkouts in sync.

---

## 0. Servers

Two GCP VMs are available. Both live in project `doc-structure`, zone `us-central1-a`.

| Alias (this doc) | VM instance name | Role | Status |
|------------------|-----------------|------|--------|
| **`doc-structure`** | `doc-structure` | Primary — fully provisioned, has rule pool, eval results, secrets, FinanceBench processed docs | Active |
| **`lsf`** | `lsf` | Secondary — fresh Debian 12 VM; bootstrap pending (no git / pip / `~/LSF`) | Standby |

Pick the server you want by substituting the alias into the `gcloud compute ssh <SERVER>` template throughout this doc. Examples below use `doc-structure` by default; for `lsf` replace the instance name.

```bash
# Choose the server at the top of your shell session
SERVER=doc-structure       # or:  SERVER=lsf
```

Then later commands become `gcloud compute ssh "$SERVER" ...`.

---

## 1. Sync model

Local and server(s) share state through GitHub.

- **Repo:** `https://github.com/yiminl18/LSF.git`
- **Branch:** `yiming-dev`

The canonical workflow is: edit locally &rarr; commit &rarr; push to `yiming-dev` &rarr; `git pull` on the server &rarr; run jobs &rarr; either push results from the server or `gcloud scp` them back.

```bash
# Local: make changes, push to yiming-dev
git add -A
git commit -m "..."
git push origin yiming-dev

# Server: pull the latest
git pull origin yiming-dev
```

Large result files (e.g. `results/*.json`) can either be committed and `git pull`-ed back, or copied via `gcloud scp` (see &sect;6). For very large outputs prefer `scp` to keep the repo small.

---

## 2. Prerequisites

The `gcloud` CLI is installed via Homebrew but is not on `PATH` by default. Add it:

```bash
export PATH=/opt/homebrew/share/google-cloud-sdk/bin:"$PATH"
```

To make this permanent, append the same line to `~/.zshrc` (or `~/.bashrc`).

```bash
echo 'export PATH=/opt/homebrew/share/google-cloud-sdk/bin:"$PATH"' >> ~/.zshrc
source ~/.zshrc
```

Verify:

```bash
gcloud --version
```

---

## 3. Authenticate (once per session)

If the access token has expired you'll see a `401` / `reauth required` error from any `gcloud` call. Re-authenticate:

```bash
gcloud auth login
```

A browser window will open; complete the OAuth flow. The token is cached on disk for subsequent commands.

---

## 4. SSH into the server

Interactive SSH session (drops you at a shell on the box). Replace `doc-structure` with `lsf` to target the secondary VM.

```bash
# Primary server (doc-structure)
gcloud compute ssh doc-structure \
    --zone=us-central1-a \
    --project=doc-structure \
    --tunnel-through-iap

# Secondary server (lsf)
gcloud compute ssh lsf \
    --zone=us-central1-a \
    --project=doc-structure \
    --tunnel-through-iap
```

Server-side, the project lives at `~/LSF/` (on `doc-structure`; not yet bootstrapped on `lsf`).

---

## 5. Run a command remotely (no interactive shell)

Examples below target `doc-structure`. For `lsf` just swap the instance name; everything else (project, zone, IAP flag) is identical.

```bash
gcloud compute ssh doc-structure \
    --zone=us-central1-a \
    --project=doc-structure \
    --tunnel-through-iap \
    --command="<your command>"
```

### Run a script in the background

Stays running after the SSH session ends. Captures stdout/stderr to a log file under `~/LSF/logs/`:

```bash
gcloud compute ssh doc-structure \
    --zone=us-central1-a \
    --project=doc-structure \
    --tunnel-through-iap \
    --command="cd ~/LSF && nohup python3 <script> > logs/<name>.log 2>&1 & echo PID:\$!"
```

Note the escaped `\$!` — the `$` must be passed through to the remote shell, so it is escaped on the local side.

### Check the log

```bash
gcloud compute ssh doc-structure \
    --zone=us-central1-a \
    --project=doc-structure \
    --tunnel-through-iap \
    --command="tail -30 ~/LSF/logs/<name>.log"
```

Swap `tail -30` for `tail -f` if you SSH interactively and want to follow the log live.

---

## 6. Copy results back to local

Two options.

### Option A — `gcloud scp` (large or untracked files)

```bash
gcloud compute scp --recurse \
    doc-structure:~/LSF/results/<folder> \
    /Users/yiminglin/Documents/Codebase/LSF/results/ \
    --zone=us-central1-a \
    --project=doc-structure \
    --tunnel-through-iap
```

Replace `<folder>` with the specific subdirectory (e.g. `financebench_single_cluster/llm/gpt54/one_shot/eval_merge`). The trailing slash on the destination keeps the source folder name intact under the local `results/` directory.

### Option B — `git pull` (when the server commits results)

If the server-side job auto-commits and pushes to `yiming-dev`:

```bash
git pull origin yiming-dev
```

This is the cleanest path for small JSON outputs (summaries, eval files) that should be versioned. Reserve `scp` for bulky binary artefacts or anything `.gitignore`-d.

---

## 7. Typical end-to-end workflow

```bash
# 1. Local — push changes
git add -A
git commit -m "tweak selector cost threshold"
git push origin yiming-dev

# 2. Server — pull and kick off a long-running job
gcloud compute ssh doc-structure --zone=us-central1-a --project=doc-structure --tunnel-through-iap \
    --command="cd ~/LSF && git pull origin yiming-dev && nohup python3 test/run_eval_merge_sampled.py > logs/eval_merge_sampled.log 2>&1 & echo PID:\$!"

# 3. Monitor
gcloud compute ssh doc-structure --zone=us-central1-a --project=doc-structure --tunnel-through-iap \
    --command="tail -30 ~/LSF/logs/eval_merge_sampled.log"

# 4. Pull results back (pick one)
git pull origin yiming-dev                 # if the job committed outputs
# OR
gcloud compute scp --recurse \
    doc-structure:~/LSF/results/financebench_single_cluster/llm/gpt54/one_shot/eval_merge \
    /Users/yiminglin/Documents/Codebase/LSF/results/financebench_single_cluster/llm/gpt54/one_shot/ \
    --zone=us-central1-a --project=doc-structure --tunnel-through-iap
```

---

## 8. Common pitfalls

- **`gcloud: command not found`** — `PATH` is not exported in the current shell. Run the `export` line from &sect;2 or restart the terminal after editing `~/.zshrc`.
- **`Reauthentication required`** — token expired; run `gcloud auth login`.
- **IAP tunnel hangs / `connection refused`** — first SSH after a VM start can take 30–60 s; retry once.
- **`fatal: refusing to merge unrelated histories` on `git pull`** — usually means the server is on the wrong branch. SSH in and run `git checkout yiming-dev` once.
- **`nohup` job not surviving** — ensure the `&` is *inside* the quoted command and the `echo PID:\$!` confirmation is printed; without it, the shell may not background the process correctly.

---

## 9. Bootstrapping a fresh VM (the `lsf` recipe)

Recorded for when you bring up `lsf` (or any new VM) from a clean Debian 12 image. Not yet executed.

### 9.1 Prereqs on the VM

The fresh image already has: Python 3.11, `nohup`, the `yiminglin` user with passwordless sudo, the SSH IAP entry point. It is missing: `git`, `pip`, all Python packages, Claude CLI, `~/LSF`, and any secrets / processed-doc data.

### 9.2 One-shot bootstrap (run as `yiminglin` on the new VM)

```bash
# System packages
sudo apt-get update
sudo apt-get install -y git python3-pip python3-venv

# Python packages — Debian 12 needs --break-system-packages outside a venv
pip3 install --user --break-system-packages openai tiktoken

# Clone the repo (HTTPS with GitHub Personal Access Token — see §9.3)
git clone https://github.com/yiminl18/LSF.git ~/LSF
cd ~/LSF && git checkout yiming-dev
```

### 9.3 GitHub authentication

Two options. Use whichever matches the existing `doc-structure` setup.

**HTTPS + Personal Access Token (PAT)** — cache once with the credential helper:
```bash
git config --global credential.helper store
# First push or pull will prompt for username (your GitHub login) and password (paste PAT, not actual password).
# Subsequent operations read from ~/.git-credentials.
```

**SSH key** — generate a new key pair on `lsf` and add the public half to your GitHub account:
```bash
ssh-keygen -t ed25519 -C "yiminglin@lsf"
cat ~/.ssh/id_ed25519.pub          # paste into github.com/settings/keys
git -C ~/LSF remote set-url origin git@github.com:yiminl18/LSF.git
```

### 9.4 Copy secrets from `doc-structure` &rarr; `lsf`

The Azure API keys for `gpt54` / `gpt54mini` live in `~/api_keys/azure_cloudbank/` (one file per model, read by `src/models/gpt54.py` and `src/models/gpt54mini.py`). They are NOT in git. `~/LSF/local/` also holds helper scripts that are not in git.

The FinanceBench processed-doc JSONs (`data/financebench/processing/`, ~267 MB / 60 docs) **are** in git as of yiming-dev, so they come with `git clone`; no scp needed.

```bash
# Azure API keys (~16 KB)
gcloud compute scp --recurse \
    doc-structure:~/api_keys \
    lsf:~/ \
    --zone=us-central1-a --project=doc-structure --tunnel-through-iap

# Local helper scripts (~few hundred KB)
gcloud compute scp --recurse \
    doc-structure:~/LSF/local \
    lsf:~/LSF/ \
    --zone=us-central1-a --project=doc-structure --tunnel-through-iap
```

### 9.5 Install + authenticate Claude CLI (for agentic pipelines)

```bash
# Install (server-side)
curl -fsSL https://claude.ai/install.sh | bash    # or follow the latest official method

# One-time login — interactive browser flow
claude /login                                      # paste the URL into a local browser
```

A successful login creates `~/.claude/.credentials.json` matching the `doc-structure` setup. Verify with:
```bash
ls -la ~/.claude/.credentials.json
claude --version
```

### 9.6 Smoke tests after bootstrap

Confirm the new VM can run all three pipeline modes before pushing real workload to it.

```bash
cd ~/LSF
# Models load with their credentials
python3 -c "from src.models.gpt54 import client, AZURE_DEPLOYMENT; print('gpt54:', AZURE_DEPLOYMENT)"
python3 -c "from src.models.gpt54mini import client, AZURE_DEPLOYMENT; print('gpt54mini:', AZURE_DEPLOYMENT)"

# Agentic tools work
python3 tools/list_rules.py --question-slug what_is_the_registrants_telephone_number_10_llm | head -3
python3 tools/compute_cost.py --question-slug what_is_the_registrants_telephone_number_10_llm \
    --rules rule_address_line_with_phone --format text

# Repo state correct
git status
git log --oneline -3
```

If all three blocks succeed, the new VM is interchangeable with `doc-structure` for any LSF workload.

### 9.7 Disk planning

A full bootstrap leaves `~/LSF` at roughly:

| Component | Size |
|-----------|-----:|
| Git checkout (yiming-dev) | ~120 MB (rule files + checked-in results) |
| `local/` secrets | ~1 MB |
| `data/financebench/processing/` | ~500 MB |
| Per-pipeline intermediates (selector_run_*, eval intermediates) — generated during runs | up to 1 GB; clean up after each run |

The current Debian 12 image ships with a 9.7 GB root partition. Bootstrap + one or two pipeline runs fits comfortably; after several runs the `selector_run_*` and `eval_*/run_*` intermediate directories should be pruned (see &sect;6 above, or the `disk_watchdog.sh` pattern used on `doc-structure`).
