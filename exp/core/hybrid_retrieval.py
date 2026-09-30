"""BM25 + semantic retrieval with rank fusion and explicit abstention.

Uses both task input and current answer. Exact self-source exclusion and
outcome-controlled random retrieval remain enforced by the shared selector.
The same encoder cache survives online index rebuilds; only new texts encode.
"""
from __future__ import annotations

from .bm25_fields import ExperienceRetriever


class LocalSentenceEncoder:
    def __init__(self, model_path, device="cpu"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_path, device=device, local_files_only=True)
        self.cache = {}

    def encode(self, texts):
        import numpy as np
        missing = list(dict.fromkeys(t for t in texts if t not in self.cache))
        if missing:
            vectors = self.model.encode(missing, batch_size=64, normalize_embeddings=True,
                                        show_progress_bar=False, convert_to_numpy=True)
            self.cache.update(zip(missing, vectors))
        return np.asarray([self.cache[t] for t in texts])


class HybridExperienceRetriever(ExperienceRetriever):
    def __init__(self, experiences, encoder, semantic_weight=0.5,
                 min_semantic_similarity=0.2, rrf_k=60):
        import numpy as np
        super().__init__(experiences)
        if not 0 <= semantic_weight <= 1 or not -1 <= min_semantic_similarity <= 1 or rrf_k <= 0:
            raise ValueError("invalid hybrid retrieval parameters")
        self.encoder = encoder
        self.semantic_weight = semantic_weight
        self.min_semantic_similarity = min_semantic_similarity
        self.rrf_k = rrf_k
        self.inputs = np.asarray(encoder.encode([e.source_input for e in experiences]))
        self.states = np.asarray(encoder.encode([e.state_before for e in experiences]))

    def rebuild(self, experiences):
        return type(self)(experiences, self.encoder, self.semantic_weight,
                          self.min_semantic_similarity, self.rrf_k)

    def combined_scores(self, query_input, query_state, alpha):
        import numpy as np
        lexical, lexical_i, lexical_s = super().combined_scores(query_input, query_state, alpha)
        if not self.experiences:
            return {}, {}, {}
        q = np.asarray(self.encoder.encode([query_input, query_state]))
        sim_i = self.inputs @ q[0]
        sim_s = self.states @ q[1]
        semantic = alpha * sim_i + (1-alpha) * sim_s
        # No forced examples when neither lexical overlap nor semantic support exists.
        lexical = {i: lexical.get(i, 0.0) for i in range(len(self.experiences))}
        eligible = [i for i in lexical if lexical[i] > 0 or semantic[i] >= self.min_semantic_similarity]
        def ranks(values):
            return {i: rank+1 for rank, i in enumerate(sorted(
                eligible, key=lambda i: (-float(values[i]), self.experiences[i].exp_id)))}
        lr, sr = ranks(lexical), ranks(semantic)
        w, k = self.semantic_weight, self.rrf_k
        fused = {i: (1-w)/(k+lr[i]) + w/(k+sr[i]) for i in eligible}
        # Trace sim_input/state are cosine values for this retriever; fused exp_scores
        # are RRF scores. The optimized config records this different interpretation.
        return fused, {i: float(sim_i[i]) for i in eligible}, {i: float(sim_s[i]) for i in eligible}
