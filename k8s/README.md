# Kubernetes manifests

| File | What it runs |
|---|---|
| [`serve.yaml`](serve.yaml) | Inference Deployment (2 replicas, GPU), Service, PodDisruptionBudget, queue-depth autoscaler. Applied with `kubectl apply -k k8s/` |
| [`bench-job.yaml`](bench-job.yaml) | One run of every benchmark suite on a GPU node, results to a PVC |
| [`ddp-job.yaml`](ddp-job.yaml) | Single-node, 2-GPU data-parallel training with `torchrun` |
| [`pytorchjob.yaml`](pytorchjob.yaml) | Multi-node data-parallel training (needs the Kubeflow Training Operator) |

## Image tags

Manifests pin `ghcr.io/apayne185/qnnbench:0.1.0`, never `:latest`. A mutable tag
means two pods of one Deployment can run different code, and a rollback cannot
restore what actually ran. The Docker workflow publishes:

- `sha-<commit>` for every push to `main`: immutable; deploy these
- `main`: moves with the branch; for trying things out
- `X.Y.Z` when a `vX.Y.Z` git tag is pushed

To deploy a specific build of the serving stack:

```bash
cd k8s && kustomize edit set image ghcr.io/apayne185/qnnbench:sha-1a2b3c4
kubectl apply -k .
```

## Serving behaviour

- **Rollouts** keep full capacity (`maxUnavailable: 0`).
- **Shutdown:** `preStop` waits 5 s so the pod leaves the Service endpoints,
  then SIGTERM makes `/readyz` return 503 and the app drains its queue before
  exiting. `terminationGracePeriodSeconds` covers all three steps.
- **Overload:** uvicorn `--limit-concurrency` refuses excess connections with
  503, the bounded app queue sheds with 503 + `Retry-After`, and requests past
  their deadline get 504 without using the GPU.
- **Autoscaling** uses queue depth from `/metrics` (requires
  prometheus-adapter), not CPU, which says little about a GPU-bound service.
- **Security:** non-root, read-only root filesystem (writable `/tmp` only for
  compile caches), no privilege escalation, all capabilities dropped.

## Validation

`k8s/validate.sh` checks every manifest against the Kubernetes schemas with
`kubeconform -strict` (no cluster needed), including the PyTorchJob CRD.
CI runs it on every PR.
