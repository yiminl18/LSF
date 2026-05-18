# Sync — Local &harr; GCP Server

This document describes how to (a) connect to the GCP servers that run the LSF pipeline, and (b) keep the local checkout and the server checkouts in sync.

---

## 0. Servers

Two GCP VMs are available. Both live in project `doc-structure`, zone `us-central1-a`. **Whenever this doc says "the server", it means `lsf`.**

| Priority | VM instance name | Role | Disk | Status |
|----------|-----------------|------|------|--------|
| **`lsf`** ⭐ | `lsf` | **Default** — primary working server | 9.7 G root + **100 G `/mnt/data`** (where `~/LSF` lives) | Active, fully provisioned |
| `doc-structure` | `doc-structure` | Secondary — older box, kept for historical runs | 9.7 G root only | Active but space-constrained; prefer `lsf` for new work |

Both share the same GitHub auth and the same Azure API keys (mirrored). To target the non-default server, pass the instance name explicitly:

```bash
# Default (lsf) — these forms are interchangeable
gcloud compute ssh lsf --project=doc-structure --zone=us-central1-a --tunnel-through-iap

# Target doc-structure explicitly when you need to
gcloud compute ssh doc-structure --project=doc-structure --zone=us-central1-a --tunnel-through-iap
```

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

Interactive SSH session (drops you at a shell on `lsf`):

```bash
gcloud compute ssh lsf \
    --zone=us-central1-a \
    --project=doc-structure \
    --tunnel-through-iap
```

Server-side, the project lives at `~/LSF/` — a symlink to `/mnt/data/LSF` on the 100 GB extra disk, so there's plenty of room for intermediates.

To target the secondary `doc-structure` VM (e.g. to compare against an older run that lives there), swap the instance name:

```bash
gcloud compute ssh doc-structure --zone=us-central1-a --project=doc-structure --tunnel-through-iap
```

---

## 5. Run a command remotely (no interactive shell)

Default target is `lsf`. Swap the instance name to target `doc-structure`.

```bash
gcloud compute ssh lsf \
    --zone=us-central1-a \
    --project=doc-structure \
    --tunnel-through-iap \
    --command="<your command>"
```

### Run a script in the background

Stays running after the SSH session ends. Captures stdout/stderr to a log file under `~/LSF/logs/`:

```bash
gcloud compute ssh lsf \
    --zone=us-central1-a \
    --project=doc-structure \
    --tunnel-through-iap \
    --command="cd ~/LSF && nohup python3 <script> > logs/<name>.log 2>&1 & echo PID:\$!"
```

Note the escaped `\$!` — the `$` must be passed through to the remote shell, so it is escaped on the local side.

### Check the log

```bash
gcloud compute ssh lsf \
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
    lsf:~/LSF/results/<folder> \
    /Users/yiminglin/Documents/Codebase/LSF/results/ \
    --zone=us-central1-a \
    --project=doc-structure \
    --tunnel-through-iap
```

Replace `<folder>` with the specific subdirectory (e.g. `financebench_single_cluster/llm/gpt54/one_shot/eval_merge`). The trailing slash on the destination keeps the source folder name intact under the local `results/` directory. Substitute `doc-structure:` for `lsf:` to pull from the secondary VM instead.

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

# 2. Server (lsf) — pull and kick off a long-running job
gcloud compute ssh lsf --zone=us-central1-a --project=doc-structure --tunnel-through-iap \
    --command="cd ~/LSF && git pull origin yiming-dev && nohup python3 test/run_eval_merge_sampled.py > logs/eval_merge_sampled.log 2>&1 & echo PID:\$!"

# 3. Monitor
gcloud compute ssh lsf --zone=us-central1-a --project=doc-structure --tunnel-through-iap \
    --command="tail -30 ~/LSF/logs/eval_merge_sampled.log"

# 4. Pull results back (pick one)
git pull origin yiming-dev                 # if the job committed outputs
# OR
gcloud compute scp --recurse \
    lsf:~/LSF/results/financebench_single_cluster/llm/gpt54/one_shot/eval_merge \
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

## 9. Bootstrapping a fresh VM (the `lsf` recipe — completed for `lsf` on 2026-05-18)

This is the recipe used to bootstrap `lsf` from a clean Debian 12 image, and the recipe to use for any future fresh VM. Each step is annotated with the actual `lsf` provisioning outcome where applicable.

### 9.1 Prereqs on a fresh VM

A clean Debian 12 image typically has: Python 3.11, `nohup`, the `yiminglin` user with passwordless sudo, the SSH IAP entry point. It is missing: `git`, `pip`, all Python packages, Claude CLI, `~/LSF`, secrets, and (if applicable) the extra data disk is not mounted.

### 9.2 Mount the extra disk (do this **before** anything else if the VM has a separate data volume)

`lsf` ships with a 9.7 GB root partition (`/dev/sda1`) and a 100 GB extra disk (`/dev/sdb`) that is unformatted and unmounted out of the box. Format, mount, and symlink `~/LSF` to the big disk so all heavy intermediates (logs, eval merge runs, agent traces, selector_run dirs) land there instead of fighting with the OS for space.

```bash
# Format /dev/sdb as ext4 (destructive — confirm the device name first with lsblk!)
sudo mkfs.ext4 -F -L lsf-data /dev/sdb

# Mount at /mnt/data, give ownership to your user
sudo mkdir -p /mnt/data
sudo mount /dev/sdb /mnt/data
sudo chown $USER:$USER /mnt/data

# Persist across reboot via fstab (nofail = boot proceeds even if disk missing)
echo "/dev/sdb  /mnt/data  ext4  defaults,nofail  0  2" | sudo tee -a /etc/fstab

# Symlink so ~/LSF transparently lives on the big disk
mkdir /mnt/data/LSF
ln -s /mnt/data/LSF ~/LSF
```

After this step, `~/LSF` is a symlink to `/mnt/data/LSF` with ~93 GB free. Scripts and `sync.md` use `~/LSF` unchanged.

### 9.3 System + Python packages (run as `yiminglin` on the new VM)

```bash
# System packages
sudo apt-get update
sudo apt-get install -y git python3-pip python3-venv

# Python packages — Debian 12 needs --break-system-packages outside a venv
pip3 install --user --break-system-packages openai tiktoken
```

### 9.4 GitHub authentication

Two options. The path used for `lsf` was option B (mirror the existing SSH key from `doc-structure`).

**Option A — generate a new SSH key on the VM and add to GitHub:**
```bash
ssh-keygen -t ed25519 -C "yiminglin@<vm-name>"
cat ~/.ssh/id_ed25519.pub          # paste into github.com/settings/keys
# Then add this to ~/.ssh/config:
#   Host github.com
#     IdentityFile ~/.ssh/id_ed25519
#     IdentitiesOnly yes
```

**Option B — copy the existing key from another provisioned VM** (the `lsf` path):
```bash
# From your local machine, with both VMs reachable:
TMPDIR=$(mktemp -d)
gcloud compute scp --project=doc-structure --zone=us-central1-a --tunnel-through-iap \
    doc-structure:~/.ssh/github_ed25519 \
    doc-structure:~/.ssh/github_ed25519.pub \
    "$TMPDIR/"
gcloud compute scp --project=doc-structure --zone=us-central1-a --tunnel-through-iap \
    "$TMPDIR/github_ed25519" "$TMPDIR/github_ed25519.pub" \
    lsf:~/.ssh/
rm -rf "$TMPDIR"

# Then on the new VM:
chmod 600 ~/.ssh/github_ed25519
chmod 644 ~/.ssh/github_ed25519.pub
cat >> ~/.ssh/config <<EOF
Host github.com
  HostName github.com
  User git
  IdentityFile ~/.ssh/github_ed25519
  IdentitiesOnly yes
  StrictHostKeyChecking accept-new
EOF
chmod 600 ~/.ssh/config

# Verify
ssh -T git@github.com   # should print: "Hi yiminl18! You've successfully authenticated..."
```

### 9.5 Clone the repo

```bash
# If ~/LSF is a symlink (after §9.2), the symlink target dir must be empty or non-existent:
rmdir /mnt/data/LSF 2>/dev/null
git clone --branch yiming-dev git@github.com:yiminl18/LSF.git /mnt/data/LSF
```

### 9.6 Copy secrets from an already-provisioned VM

The Azure API keys for `gpt54` / `gpt54mini` live in `~/api_keys/azure_cloudbank/` (one file per model, read by `src/models/gpt54.py` and `src/models/gpt54mini.py`). They are NOT in git. `~/LSF/local/` also holds helper scripts that are not in git.

The FinanceBench processed-doc JSONs (`data/financebench/processing/`, ~267 MB / 60 docs) **are** in git as of yiming-dev, so they come with `git clone`; no scp needed.

```bash
# Azure API keys (~16 KB) — from local machine
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

### 9.7 Install + authenticate Claude CLI (only if you need agentic pipelines)

The algorithmic pipelines (Pareto v2, fallback, v1, etc.) only need gpt54 access and DO NOT require Claude CLI. Install only if you want to run agentic rule generation or selection.

```bash
# Install (server-side)
curl -fsSL https://claude.ai/install.sh | bash    # or follow the latest official method

# One-time login — interactive browser flow
claude /login                                      # paste the URL into a local browser
```

A successful login creates `~/.claude/.credentials.json`. Verify with:
```bash
ls -la ~/.claude/.credentials.json
claude --version
```

### 9.8 Smoke tests after bootstrap

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

If all three blocks succeed, the new VM is interchangeable with the others for any LSF workload.

### 9.9 Disk planning

After bootstrap, with everything in place, `~/LSF` totals roughly:

| Component | Size |
|-----------|-----:|
| Git checkout (yiming-dev) | ~330 MB (rule files + checked-in results + processed-doc JSONs) |
| `local/` helper scripts | ~1 MB |
| `~/api_keys/` Azure keys | ~16 KB |
| Per-pipeline intermediates (selector_run_*, eval_*/run_*) — generated during runs | up to 1 GB per run; clean up after |

**On `lsf` (default):** ~330 MB checkout lives on the 100 GB `/mnt/data` partition (via the `~/LSF → /mnt/data/LSF` symlink from §9.2). The 9.7 GB root partition stays for OS + pip cache + `~/.claude/`. Intermediates can accumulate freely without disk pressure.

**On a VM without an extra disk** (e.g., `doc-structure`): the root partition holds everything. Periodic cleanup of `selector_run_*` and `eval_*/run_*` is required; see `disk_watchdog.sh` for an automated pattern.
