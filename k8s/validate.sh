#!/usr/bin/env bash
# Validate every manifest against the Kubernetes API schemas, offline from any cluster.
#   KUBECONFORM=/path/to/kubeconform k8s/validate.sh
# PyTorchJob is a Kubeflow CRD with no published JSON schema, so its schema is
# extracted from the Training Operator's released CRD.
set -euo pipefail
cd "$(dirname "$0")"
KUBECONFORM=${KUBECONFORM:-kubeconform}
CRD_URL=https://raw.githubusercontent.com/kubeflow/training-operator/v1.8.1/manifests/base/crds/kubeflow.org_pytorchjobs.yaml

schemas=$(mktemp -d)
trap 'rm -rf "$schemas"' EXIT
mkdir -p "$schemas/kubeflow.org"
curl -fsSL "$CRD_URL" | python3 -c '
import json, sys, yaml
crd = yaml.safe_load(sys.stdin)
for v in crd["spec"]["versions"]:
    schema = v["schema"]["openAPIV3Schema"]
    with open(sys.argv[1] + "/kubeflow.org/pytorchjob_" + v["name"] + ".json", "w") as f:
        json.dump(schema, f)
' "$schemas"

"$KUBECONFORM" -strict -summary \
  -schema-location default \
  -schema-location "$schemas/{{.Group}}/{{.ResourceKind}}_{{.ResourceAPIVersion}}.json" \
  bench-job.yaml ddp-job.yaml pytorchjob.yaml serve.yaml
# The kustomized serving stack, as `kubectl apply -k` would send it.
kubectl kustomize . | "$KUBECONFORM" -strict -summary -
