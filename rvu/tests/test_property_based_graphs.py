import pytest

try:
    from hypothesis import given
    from hypothesis import strategies as st
except ModuleNotFoundError:  # pragma: no cover
    pytest.skip("hypothesis not installed", allow_module_level=True)

from rvu.provenance.closure import contamination_closure
from rvu.provenance.dag import ArtifactKind, ArtifactNode, Edge, EdgeKind, Principal, ProvenanceDAG


@st.composite
def dag_and_seed(draw: st.DrawFn) -> tuple[ProvenanceDAG, set[str]]:
    size = draw(st.integers(min_value=2, max_value=8))
    ids = [f"n{i}" for i in range(size)]
    dag = ProvenanceDAG()
    for nid in ids:
        dag.add_node(ArtifactNode(nid, ArtifactKind.INPUT, Principal.USER, {}))
    for i in range(size - 1):
        if draw(st.booleans()):
            dag.add_edge(Edge(ids[i], ids[i + 1], EdgeKind.DERIVED_FROM))
    seed = {draw(st.sampled_from(ids))}
    return dag, seed


@given(dag_and_seed())
def test_closure_contains_seed(data: tuple[ProvenanceDAG, set[str]]) -> None:
    dag, seed = data
    closure = contamination_closure(dag, seed)
    assert seed.issubset(closure)


@given(dag_and_seed())
def test_closure_monotonic(data: tuple[ProvenanceDAG, set[str]]) -> None:
    dag, seed = data
    bigger = set(seed)
    bigger.update(dag.nodes.keys())
    assert contamination_closure(dag, seed).issubset(contamination_closure(dag, bigger))
