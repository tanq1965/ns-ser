"""Knowledge graph construction (docs/ontology_design.md, section 2).

Builds the bipartite fact-rule graph:
    concept nodes --premise--> rule node --conclusion--> concept/intermediate/emotion node
from knowledge/ontology.yaml + knowledge/rules/*.yaml, in memory, once, at startup.
"""

import glob
import os

import networkx as nx
import yaml

KnowledgeGraph = nx.DiGraph


def load_kg(path: str) -> KnowledgeGraph:
    """Load ontology + rules under `path` (a knowledge/ directory) into a bipartite fact-rule graph.

    Reads `<path>/ontology.yaml` for concept/emotion nodes and every
    `<path>/rules/*.yaml` for rule nodes, per docs/ontology_design.md.
    """
    ontology_path = os.path.join(path, "ontology.yaml")
    rules_dir = os.path.join(path, "rules")

    with open(ontology_path, "r", encoding="utf-8") as f:
        ontology = yaml.safe_load(f)

    graph = KnowledgeGraph()

    for concept in ontology.get("acoustic_concepts", []):
        graph.add_node(concept, kind="acoustic_concept")
    for concept in ontology.get("intermediate_concepts", []):
        graph.add_node(concept, kind="intermediate_concept")
    for emotion in ontology.get("emotions", []):
        graph.add_node(emotion, kind="emotion")

    rule_files = sorted(glob.glob(os.path.join(rules_dir, "*.yaml")))
    for rule_file in rule_files:
        with open(rule_file, "r", encoding="utf-8") as f:
            rules = yaml.safe_load(f) or []
        for rule in rules:
            rule_id = rule["id"]
            graph.add_node(
                rule_id,
                kind="rule",
                type=rule["type"],
                min_matches=rule.get("min_matches"),
                conclusion=rule["conclusion"],
                explanation=rule["explanation"],
            )
            for concept in rule["inputs"]:
                graph.add_edge(concept, rule_id, relation="premise")
            graph.add_edge(rule_id, rule["conclusion"], relation="conclusion")

    return graph
