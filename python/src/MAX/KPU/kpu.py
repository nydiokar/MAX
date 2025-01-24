from typing import Dict, Any, List, Optional
from MAX.retrievers.kb_retriever import KnowledgeBasesRetriever
from transformers import AutoTokenizer, AutoModel, pipeline
import torch
import numpy as np


class KPU:
    """
    Knowledge Processing Unit (KPU) - Responsible for context enrichment,
    knowledge verification, and request analysis.

    Typical Flow:
    - Retrieve additional context from a knowledge base (if available).
    - Analyze the user input to suggest task breakdowns, identify collaborators,
      and highlight where feedback is needed.
    - Verify claims against knowledge stored in the knowledge base.
    """

    def __init__(self, retriever: Optional[KnowledgeBasesRetriever] = None):
        """
        Initialize the KPU with an optional knowledge bases retriever.

        :param retriever: An instance of KnowledgeBasesRetriever to fetch relevant context.
        """
        self.retriever = retriever
        self.verifier = None  # Optional verifier if needed later.

        # Load language models and pipelines
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        self.model = AutoModel.from_pretrained(
            "roberta-base", output_attentions=True
        )
        self.similarity_model = pipeline(
            "feature-extraction",
            model="sentence-transformers/all-MiniLM-L6-v2",
            tokenizer="sentence-transformers/all-MiniLM-L6-v2",
        )

    async def process_request(
        self,
        user_input: str,
        selected_agent: str,
        conversation_history: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """
        Enrich the request and return processed information.

        Steps:
        1. Retrieve additional context from the knowledge base (if available).
        2. Analyze the request in the context of the selected agent and retrieved info.
        3. Verify extracted claims against stored knowledge.
        4. Return enriched context, decompositions, collaborators, verification results, and feedback points.

        :param user_input: The user's query string.
        :param selected_agent: The agent type/identifier.
        :param conversation_history: The history of conversation so far.
        :return: Dict with keys:
                 - enriched_context
                 - suggested_decomposition
                 - collaborators
                 - verification_results
                 - feedback_required
                 - conversation_history
        """
        # Retrieve context if retriever is available
        context = ""
        if self.retriever:
            context = await self.retriever.retrieve_and_combine_results(
                user_input, agent_type=selected_agent
            )

        analysis = await self._analyze_request(
            user_input, selected_agent, context
        )
        verification = await self._verify_knowledge(analysis)

        return {
            "enriched_context": context,
            "suggested_decomposition": analysis.get("task_breakdown", []),
            "collaborators": analysis.get("suggested_collaborators", []),
            "verification_results": verification,
            "feedback_required": analysis.get("feedback_points", []),
            "conversation_history": conversation_history,
        }

    async def _analyze_request(
        self, user_input: str, selected_agent: str, context: str
    ) -> Dict[str, Any]:
        """
        Analyze the request to propose a task breakdown, identify collaborators, and highlight feedback needs.

        :param user_input: The user's input.
        :param selected_agent: The agent type/identifier.
        :param context: The enriched context from the knowledge base.
        :return: Dict with 'task_breakdown', 'suggested_collaborators', and 'feedback_points'.
        """
        analysis = {
            "task_breakdown": [],
            "suggested_collaborators": [],
            "feedback_points": [],
        }

        # Attempt to break down tasks if a retriever is available
        if self.retriever:
            analysis["task_breakdown"].extend(
                await self._derive_task_breakdown(user_input, selected_agent)
            )

        # Identify potential collaborators
        analysis["suggested_collaborators"].extend(
            self._identify_collaborators(user_input, context)
        )

        # Identify points where feedback is needed
        analysis["feedback_points"].extend(
            self._identify_feedback_points(user_input)
        )

        return analysis

    async def _derive_task_breakdown(
        self, user_input: str, selected_agent: str
    ) -> List[str]:
        """
        Derive a breakdown of tasks by retrieving patterns associated with the selected agent.

        :param user_input: The user's input.
        :param selected_agent: The agent type/identifier.
        :return: List of subtasks derived from known task patterns.
        """
        subtasks = []
        task_patterns = await self.retriever.retrieve(
            f"task patterns for {selected_agent}",
            filters={"type": "task_pattern", "agent": selected_agent},
        )

        for pattern in task_patterns:
            # If pattern content matches the user input, use its subtasks
            if pattern["content"]["text"] in user_input:
                subtasks.extend(pattern["metadata"].get("subtasks", []))
        return subtasks

    def _identify_collaborators(
        self, user_input: str, context: str
    ) -> List[Dict[str, Any]]:
        """
        Identify potential collaborators from the user input or context.

        :param user_input: The user's input.
        :param context: The enriched context.
        :return: A list of collaborator descriptors.
        """
        collaboration_triggers = [
            "need data from",
            "requires input",
            "collaborate with",
            "verify with",
        ]

        collaborators = []
        combined_text = f"{user_input}\n{context}".lower()
        for trigger in collaboration_triggers:
            if trigger in combined_text:
                collaborators.append(
                    {
                        "type": (
                            "agent" if "agent" in combined_text else "human"
                        ),
                        "reason": f"Triggered by: {trigger}",
                        "confidence": 0.8,
                    }
                )
        return collaborators

    def _identify_feedback_points(
        self, user_input: str
    ) -> List[Dict[str, Any]]:
        """
        Identify where user feedback or clarification is needed.

        :param user_input: The user's input.
        :return: A list of feedback points.
        """
        feedback_triggers = [
            "confirm",
            "specify",
            "choose between",
            "prefer",
            "clarify",
        ]
        feedback_points = []

        lower_input = user_input.lower()
        for trigger in feedback_triggers:
            if trigger in lower_input:
                feedback_points.append(
                    {
                        "type": (
                            "confirmation"
                            if trigger == "confirm"
                            else "clarification"
                        ),
                        "point": f"Need to {trigger}: {user_input}",
                        "priority": (
                            "high" if trigger == "confirm" else "medium"
                        ),
                    }
                )
        return feedback_points

    async def _verify_knowledge(
        self, analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Verify claims found in the analysis against the knowledge base.

        :param analysis: The analysis dict from _analyze_request().
        :return: Dict with 'verified_facts', 'uncertain_claims', and 'contradictions'.
        """
        results = {
            "verified_facts": [],
            "uncertain_claims": [],
            "contradictions": [],
        }

        # If no retriever, can't verify
        if not self.retriever:
            return results

        # Extract claims from the task breakdown
        claims = []
        for task in analysis.get("task_breakdown", []):
            claims.extend(self._extract_claims(task))

        # Verify each claim
        for claim in claims:
            claim_result = await self._verify_claim(claim)
            # Merge claim_result into results
            for key, value in claim_result.items():
                results[key].extend(value)

        return results

    async def _verify_claim(
        self, claim: str
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Verify a single claim against the knowledge base.

        :param claim: The claim to verify.
        :return: Dict with lists under 'verified_facts', 'uncertain_claims', 'contradictions'.
        """
        result = {
            "verified_facts": [],
            "uncertain_claims": [],
            "contradictions": [],
        }

        facts = await self.retriever.retrieve(claim, filters={"type": "fact"})
        if not facts:
            result["uncertain_claims"].append(
                {"claim": claim, "reason": "No supporting evidence found"}
            )
            return result

        supporting_facts = []
        contradictions = []
        for fact in facts:
            fact_text = fact["content"]["text"]
            similarity = self._compute_semantic_similarity(claim, fact_text)
            if similarity > 0.8:
                # Check contradictions
                if self._is_contradiction(claim, fact_text):
                    contradictions.append(fact_text)
                else:
                    supporting_facts.append(fact_text)

        if contradictions:
            result["contradictions"].append(
                {"claim": claim, "contradicting_facts": contradictions}
            )
        elif supporting_facts:
            result["verified_facts"].append(
                {"claim": claim, "supporting_facts": supporting_facts}
            )
        else:
            result["uncertain_claims"].append(
                {"claim": claim, "reason": "Insufficient evidence"}
            )

        return result

    def _extract_claims(self, text: str) -> List[str]:
        """
        Extract claims (factual sentences) from the given text.

        :param text: The text from which to extract claims.
        :return: A list of sentence strings considered as claims.
        """
        inputs = self.tokenizer(
            text, return_tensors="pt", max_length=512, truncation=True
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
            hidden_states = outputs.last_hidden_state

        token_importance = torch.norm(hidden_states, dim=-1).squeeze(0)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        # Simple sentence splitting
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        claims = []
        current_pos = 0

        for sent in sentences:
            sent_tokens = self.tokenizer(
                sent, return_tensors="pt", truncation=True
            )
            sent_length = len(sent_tokens["input_ids"][0])

            # Check if we have enough tokens left
            if current_pos + sent_length <= len(token_importance):
                sent_importance = token_importance[
                    current_pos : current_pos + sent_length
                ].mean()
                # A very simple heuristic to pick out "factual" sentences
                if sent_importance > 0.5 or any(
                    keyword in sent.lower()
                    for keyword in [
                        "is",
                        "are",
                        "has",
                        "have",
                        "should",
                        "must",
                    ]
                ):
                    claims.append(sent)
                current_pos += sent_length

        return claims

    def _compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Compute semantic similarity using MiniLM embeddings and cosine similarity.

        :param text1: First text string.
        :param text2: Second text string.
        :return: A similarity score (0.0 to 1.0).
        """
        emb1 = self.similarity_model(text1)
        emb2 = self.similarity_model(text2)

        emb1 = np.mean(emb1[0], axis=0)
        emb2 = np.mean(emb2[0], axis=0)

        similarity = np.dot(emb1, emb2) / (
            np.linalg.norm(emb1) * np.linalg.norm(emb2)
        )
        return float(similarity)

    def _is_contradiction(self, claim: str, fact: str) -> bool:
        """
        Check for contradictions between the claim and the fact.
        This uses a simple similarity-based heuristic as a placeholder.

        :param claim: The claim text.
        :param fact: The fact text.
        :return: True if considered contradictory, False otherwise.
        """
        inputs = self.tokenizer(
            claim, fact, return_tensors="pt", padding=True, truncation=True
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
            pooled_output = outputs.last_hidden_state[:, 0, :]

        similarity = torch.cosine_similarity(
            pooled_output[0], pooled_output[1], dim=0
        )
        return float(similarity) < 0.3

    def verify_response(self, agent_response: str) -> bool:
        """
        Perform a simple verification step on the agent's response.

        :param agent_response: The response from the agent.
        :return: True if coherent or if no verifier is set, False otherwise.
        """
        if self.verifier:
            return self.verifier.verify(agent_response)
        return True
