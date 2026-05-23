from typing import Callable

import jax
import jax.numpy as jnp

from .attention_ops import AttentionOperatorMixin
from .regularization_metrics import RegularizationMetricMixin
from .statistics import TopicStatisticsMixin
from .training import TrainingMixin


class AttentiveTopicModel(
    TrainingMixin,
    TopicStatisticsMixin,
    AttentionOperatorMixin,
    RegularizationMetricMixin,
):
    """
    Topic model with a local topic-specific attention operator.

    Dimension notation used by the implementation:
    - ``V``: vocabulary size, ``vocab_size``;
    - ``T``: number of topics, ``n_topics``;
    - ``L``: context radius, ``ctx_len``;
    - ``K = 2 * L + 1``: attention kernel length;
    - ``N``: number of tokens in the current token sequence;
    - ``D``: number of documents represented by ``ctx_bounds``.

    Core tensors:
    - ``data`` / ``batch``: ``(N,)`` token ids;
    - ``ctx_bounds``: ``(D + 1,)`` document boundary offsets;
    - ``phi``: ``(V, T)``, word-topic probabilities;
    - ``p_ti``: ``(N, T)``, token-topic probabilities;
    - ``theta_ti``: ``(N, T)``, attention-smoothed token-topic probabilities;
    - ``weights`` / ``weights_t``: ``(K, T)``, topic-specific attention kernel;
    - ``n_w``: ``(V,)``, word counts / priors;
    - ``n_t``: ``(T,)``, topic counts / priors.
    """

    def __init__(
            self,
            vocab_size: int,
            ctx_len: int,
            *,
            n_topics: int = 10,
            clip_context: str = 'default',
            optimize_weights: bool = True,
            max_self_score: float | None = None,
            use_big_n: bool = True,
            weights: jax.Array | None = None,
            n_attention_passes: int = 1,
            regularizers: list = None,
            alpha_regularizers: list[Callable[[jax.Array], jax.Array | float]] | None = None,
            metrics: list = None,
            eps: float = 1e-12,
    ):
        """
        Args:
            vocab_size: ``V``.
            ctx_len: ``L``.
            n_topics: ``T``.
            weights: optional initial attention kernel, shape ``(K, T)``.
            regularizers: regularizers over ``phi`` with shape ``(V, T)``.
            alpha_regularizers: regularizers over ``weights_t`` with shape ``(K, T)``.
            metrics: metrics receiving ``phi_it`` ``(N, T)``, ``phi_wt`` ``(V, T)``,
                and ``theta`` ``(N, T)``.
        """
        self.vocab_size = vocab_size
        self.ctx_len = ctx_len
        self.n_topics = n_topics
        self.clip_context = clip_context
        self.optimize_weights = optimize_weights
        self.max_self_score = max_self_score
        self.use_big_n = use_big_n
        self.weights = None if weights is None else jnp.asarray(weights)
        self.n_attention_passes = n_attention_passes
        self._eps = eps
        self.phi = None
        self.n_t = None
        self.n_w = None
        self._weights_update_locked = False

        self._regularizations = {}
        if regularizers is not None:
            for regularization in regularizers:
                self.add_regularization(regularization)

        self._alpha_regularizations = {}
        if alpha_regularizers is not None:
            for idx, regularization in enumerate(alpha_regularizers):
                self.add_alpha_regularization(tag=f'alpha_reg_{idx}', regularization=regularization)

        self._metrics = {}
        if metrics is not None:
            for metric in metrics:
                self.add_metric(metric)
