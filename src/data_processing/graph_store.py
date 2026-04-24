# src/data_processing/graph_store.py
import os
import hashlib
import json
from typing import cast, LiteralString
from neo4j import GraphDatabase, Query
from schema import ClinicalFact, ClinicalLogic

class Neo4jGraphStore:
  def __init__(self):
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")

    assert user and password and uri, "Missing Neo4j environment variables!"
    
    self.driver = GraphDatabase.driver(uri, auth=(user, password))

  def close(self):
    self.driver.close()
      
  def clear_database(self):
    """Wipes the entire graph. Use with caution at the start of ingestion."""
    with self.driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")

  def add_fact(self, fact: ClinicalFact):
    """
    Add a ClinicalFact to the graph db by translating it into a Cypher MERGE query.
    Example:
      MERGE (s:Measurement {name:"CHA2DS2-VASc score"})
      MERGE (o:Condition {name:"Atrial Fibrillation"})
      MERGE (s)-[:ASSESSES_RISK_OF]->(o)
    """
    s_label = fact.subject_domain.replace(" ", "")
    o_label = fact.object_domain.replace(" ", "")

    with self.driver.session() as session:
        query_str = cast(LiteralString, (
            f"MERGE (s:{s_label} {{name: $subject}})"
            f"MERGE (o:{o_label} {{name: $object}})"
            f"MERGE (s)-[:{fact.predicate.upper()}]->(o)"
        ))
        session.run(query_str, subject=fact.subject, object=fact.object)

  def add_logic(self, logic: ClinicalLogic):
    with self.driver.session() as session:
      session.execute_write(self._create_logic_tx, logic)

  @staticmethod
  def _create_logic_tx(tx, logic):
    # Create the central rule node, uniquely identified by its action, logic gate and triggers
    trigger_data = sorted([f"{t.entity}_{t.negated}" for t in logic.triggers])
    rule_string = f"{logic.action}_{logic.logic_gate}_{json.dumps(trigger_data)}"
    logic_id = hashlib.md5(rule_string.encode()).hexdigest()
    
    tx.run(
      "MERGE (l:ClinicalLogic {id: $id}) "
      "SET l.action = $action, l.gate = $gate",
      id=logic_id, 
      action=logic.action, 
      gate=logic.logic_gate
    )

    # Connect the triggers, i.e. logic requirements
    for trigger in logic.triggers:
      trigger_label = trigger.domain.replace(" ", "")

      query = (
        f"MERGE (l:ClinicalLogic {{id: $id}}) "
        f"MERGE (e:{trigger_label} {{name: $entity}}) "
        "MERGE (e)-[r:TRIGGERS]->(l) "
        "SET r.negated = $negated, r.description = $description"
      )

      tx.run(
        query,
        id=logic_id,
        entity=trigger.entity,
        negated=trigger.negated,
        description=trigger.description or ""
      )