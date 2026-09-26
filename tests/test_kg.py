from knowledge.kg import load_kg


def test_rule_nodes_have_indegree_at_least_1_and_outdegree_1():
    kg = load_kg("knowledge")
    rule_nodes = [n for n, d in kg.nodes(data=True) if d.get("kind") == "rule"]
    assert rule_nodes, "expected at least one rule node"

    for node in rule_nodes:
        assert kg.in_degree(node) >= 1
        assert kg.out_degree(node) == 1


def test_premise_edges_point_from_concept_into_rule():
    kg = load_kg("knowledge")
    premise_edges = [(u, v) for u, v, d in kg.edges(data=True) if d.get("relation") == "premise"]
    assert premise_edges, "expected at least one premise edge"

    for u, v in premise_edges:
        assert kg.nodes[u]["kind"] != "rule"
        assert kg.nodes[v]["kind"] == "rule"


def test_conclusion_edges_point_from_rule_to_concept():
    kg = load_kg("knowledge")
    conclusion_edges = [(u, v) for u, v, d in kg.edges(data=True) if d.get("relation") == "conclusion"]
    assert conclusion_edges, "expected at least one conclusion edge"

    for u, v in conclusion_edges:
        assert kg.nodes[u]["kind"] == "rule"
        assert kg.nodes[v]["kind"] != "rule"


def test_every_edge_has_a_known_relation():
    kg = load_kg("knowledge")
    for _, _, data in kg.edges(data=True):
        assert data.get("relation") in ("premise", "conclusion")
