# ALFWorld resource fix (2026-09-25)

Confirmed initialization failure: OSError errno 28, copying Fast Downward's 43 MiB
shared library into /tmp. TextWorld 1.7 PddlEnv inherits a no-op close; deleted
library mappings remained alive across task resets. One original worker had 177
distinct mapped copies. The original errors and experiment artifacts are retained.

AlfWorldEnvironment now collects planner owners through actual wrapper attributes
before normal environment closure, then invokes fast_downward.close_lib exactly
once for each owned library, including when wrapper closure raises. Repeated close
is safe. On this installation one library mapping remains resident after the first
close; subsequent resets must not increase the count. No upstream package modified.

Resource fix only: same M0, manifests, seeds, task actions, probe/gate repetitions,
editor and scope-update rules. New protocol metadata records
alfworld-planner-close-v1. Interrupted pre-fix ALFWorld runs must not be pooled into
completed benchmark estimates. Their initialization failures are infrastructure
errors, not valid model failures.

GPU0's Qwen3.5-9B launch also now uses gpu_memory_utilization=0.32 instead of 0.66
with unchanged explicit 3 GiB KV cache. The old ratio rejected startup with 28 GiB
free because it requested 52 GiB. Other processes are never stopped.

Audit, regression and 100-reset validation artifacts:
`results/alfworld-resource-fix-20260925/`.

Reruns from identical frozen M0:
- `results/resource-fixed-qwen9b-20260925`: ALFWorld and WebShop; previous v2
  deployment failed before either worker started.
- `results/resource-fixed-qwen-20260925`: ALFWorld only, queued after existing
  WebShop and verified GPU cleanup.
- `results/resource-fixed-glm-20260925`: ALFWorld only, same queue arrangement.

All queues are detached and have no overall deadline. Each watchdog cleans up its
own server after workers finish or fail; unrelated GPU jobs remain untouched.
