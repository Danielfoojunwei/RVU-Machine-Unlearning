from rvu.provenance.closure import contamination_closure
from rvu.provenance.dag import ArtifactKind, ArtifactNode, Edge, EdgeKind, Principal, ProvenanceDAG


def test_contamination_closure_reachability() -> None:
    dag = ProvenanceDAG()
    for n in ["a", "b", "c", "d"]:
        dag.add_node(ArtifactNode(n, ArtifactKind.INPUT, Principal.USER, {}))
    dag.add_edge(Edge("a", "b", EdgeKind.DERIVED_FROM))
    dag.add_edge(Edge("b", "c", EdgeKind.DERIVED_FROM))
    dag.add_edge(Edge("d", "c", EdgeKind.DERIVED_FROM))
    assert contamination_closure(dag, {"a"}) == {"a", "b", "c"}
