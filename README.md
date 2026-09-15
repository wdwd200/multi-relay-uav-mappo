# Multi-relay UAV simulation — stage 1

This repository implements the stage-1 environment skeleton specified in
`02_仿真开发阶段1_Codex任务书.md`.  It deliberately contains no training,
graph, role, routing, or safety-controller algorithm.

## Run the repeatable checks

```powershell
py -3.13 -m selfcheck.run_selfcheck
```

The command writes machine-readable results to `artifacts/selfcheck-results.json`,
plots (SVG) to `artifacts/plots/`, and updates `阶段1环境自检报告.md`.

The runtime has no third-party dependency; this keeps the self-check runnable in
a minimal Python installation.  Its public interface is:

```python
from relay_env import RelayEnv
env = RelayEnv(num_relays=4)
obs, info = env.reset(seed=7)
obs, reward, terminated, truncated, info = env.step([(0.0, 0.0, 0.0)] * 4)
```
