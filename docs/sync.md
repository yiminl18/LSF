# Sync — Local &harr; GCP Server

This document describes how to (a) connect to the GCP server that runs the LSF pipeline, and (b) keep the local checkout and the server checkout in sync.

---

## 1. Sync model

Local and server share state through GitHub.

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

Interactive SSH session (drops you at a shell on the box):

```bash
gcloud compute ssh doc-structure \
    --zone=us-central1-a \
    --project=doc-structure \
    --tunnel-through-iap
```

Server-side, the project lives at `~/LSF/`.

---

## 5. Run a command remotely (no interactive shell)

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
