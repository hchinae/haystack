from typing import Optional, Set

from haystack.graph_retriever.question import QuestionType
from haystack.graph_retriever.triple import Triple


class Query:

    def __init__(self, question_type: QuestionType, triples: Set[Triple]):
        self.triples: Set[Triple] = triples
        self.question_type: QuestionType = question_type
        self._sparql_query: Optional[str] = None

    def get_sparql_query(self):
        if not self._sparql_query:
            self._sparql_query = self.build_sparql_query_string()
        return self._sparql_query

    def build_sparql_query_string(self) -> str:
        """
        Combine triples in one where_clause and generate a SPARQL query based on a template for one of the three question types
        """
        where_clause = self.build_where_clause()
        query = None
        if self.question_type == QuestionType.CountQuestion:
            query = f"SELECT (COUNT(  ?uri ) AS ?count_result) WHERE {{ {where_clause} }}"
            # example: SELECT (COUNT(?harry) AS ?rc) WHERE { ?harry rdfs:label "Harry Potter"@en. }
        elif self.question_type == QuestionType.BooleanQuestion:
            query = f"ASK WHERE {{ {where_clause} }}"
            # example: ASK WHERE { dbr:Hermione_Granger dbo:spouse dbr:Ron_Weasley}
        elif self.question_type == QuestionType.ListQuestion:
            query = "SELECT ?uri WHERE {{ {where_clause} }}"
            # example: SELECT ?o WHERE{ ?harry rdfs:label "Harry Potter"@en. ?family rdfs:label "family"@en. ?harry ?family ?o. }

        if not query:
            raise RuntimeError(f"QuestionType {self.question_type} unknown")

        return query

    def build_where_clause(self):
        """
        Combine triples in one where_clause
        """
        triples_text = [str(triple) for triple in self.triples]
        where_clause = ". ".join(triples_text)
        return where_clause
