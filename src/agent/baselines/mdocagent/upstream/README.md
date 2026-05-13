# MDocAgent Upstream Submodule

This directory is a placeholder for the git submodule of:
  https://github.com/aiming-lab/MDocAgent

## Pin SHA

Pin to SHA: `main` (pin a specific SHA once network access is confirmed stable).
At time of authoring, the last known good commit is on the `main` branch.

## To initialise

```bash
git submodule add https://github.com/aiming-lab/MDocAgent \
    src/agent/baselines/mdocagent/upstream
git submodule update --init src/agent/baselines/mdocagent/upstream
```

If `git submodule add` fails with SSL errors:
```bash
GIT_SSL_NO_VERIFY=true git submodule add \
    https://github.com/aiming-lab/MDocAgent \
    src/agent/baselines/mdocagent/upstream
```

Once the submodule is present, remove this README and update the adapter.
