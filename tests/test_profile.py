import json

from qnnbench.profile import run
from qnnbench.sim import statevector


def test_profile_writes_trace_with_named_blocks(tmp_path):
    meta = run(n_qubits=7, batch=4, steps=1, out=tmp_path, device="cpu")
    events = json.loads((tmp_path / "trace.json").read_text())["traceEvents"]
    names = {e.get("name", "") for e in events}
    assert "train_step" in names
    assert any(n.startswith("fused_") for n in names)
    assert (tmp_path / "summary.txt").read_text().strip()
    assert meta["config"]["n_qubits"] == 7
    assert statevector.LABELS["enabled"] is False  # labels switched back off
