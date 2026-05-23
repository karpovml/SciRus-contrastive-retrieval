from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp

# data / batch                 (N,)
# ctx_bounds                   (D + 1,)

# phi                          (V, T)
# phi[batch] / values          (N, T)

# p_ti                         (N, T)
# theta_ti                     (N, T)

# weights / weights_t          (K, T)

# n_w                          (V,)
# n_t                          (T,)

# n_tw                         (V, T)
# N_tw                         (V, T)

# p_wi                         (N,)
# p_t                          (T,)
# p_w                          (V,)


class TopicStatisticsMixin:
    def _update_p_ti(
            self,
            *,
            values: jax.Array,
            theta_ti: jax.Array,
    ) -> jax.Array:
        """Update token-topic probabilities using the current topic prior.

        Args:
            values: token-local word-topic values, usually ``phi[batch]``, shape ``(N, T)``.
            theta_ti: attention-smoothed token-topic probabilities, shape ``(N, T)``.

        Uses:
            ``self.n_t``: topic prior/counts, shape ``(T,)``.

        Returns:
            Row-normalized token-topic probabilities, shape ``(N, T)``.
        """
        p_t = self.n_t / jnp.maximum(jnp.sum(self.n_t), self._eps)
        p_ti = values * theta_ti / jnp.maximum(p_t[None, :], self._eps)
        return self._norm(p_ti.T).T

    # def _calc_p_ti(
    #         self,
    #         *,
    #         phi: jax.Array,
    #         theta_ti: jax.Array,
    #         batch: jax.Array,
    # ) -> jax.Array:
    #     phi_twi = phi[batch]
    #     return self._update_p_ti(values=phi_twi, theta_ti=theta_ti)

    @partial(jax.jit, static_argnums=0)
    def _calc_n_w(
            self,
            *,
            batch: jax.Array,
    ) -> jax.Array:
        """Count tokens by vocabulary id.

        Args:
            batch: token ids, shape ``(N,)``.

        Returns:
            Word counts, shape ``(V,)``.
        """
        return jnp.bincount(
            batch,
            length=self.vocab_size,
            minlength=self.vocab_size,
        )

    def _init_word_counts(
            self,
            data: jax.Array,
    ):
        """Initialize ``self.n_w`` from token ids.

        Args:
            data: token ids, shape ``(N,)``.

        Sets:
            ``self.n_w``: word counts, shape ``(V,)``.
        """
        self.n_w = self._calc_n_w(batch=data).astype(jnp.float32)

    def _calc_p_wi(
            self,
            *,
            phi: jax.Array,
            theta_ti: jax.Array,
            batch: jax.Array,
            p_t: jax.Array | None = None,
    ) -> jax.Array:
        """Calculate token probabilities under the current model.

        Args:
            phi: word-topic probabilities, shape ``(V, T)``.
            theta_ti: token-topic probabilities, shape ``(N, T)``.
            batch: token ids, shape ``(N,)``.
            p_t: optional topic prior, shape ``(T,)``. If omitted, ``self.n_t`` is used.

        Uses:
            ``self.n_w``: word prior/counts, shape ``(V,)``.
            ``self.n_t``: topic prior/counts, shape ``(T,)``.

        Returns:
            Token probabilities, shape ``(N,)``.
        """
        if self.n_w is None or self.n_t is None:
            raise ValueError('Model priors are not initialized.')

        if p_t is None:
            p_t = self.n_t / jnp.maximum(jnp.sum(self.n_t), self._eps)
        else:
            p_t = p_t / jnp.maximum(jnp.sum(p_t), self._eps)
        theta_ti = self._norm_rows(theta_ti)
        p_w = self.n_w / jnp.maximum(jnp.sum(self.n_w), self._eps)
        return p_w[batch] * jnp.sum(
            phi[batch] * (theta_ti / jnp.maximum(p_t[None, :], self._eps)),
            axis=1,
        )

    @partial(jax.jit, static_argnums=0)
    def _calc_n_tw_small(
            self,
            *,
            p_ti: jax.Array,
            batch: jax.Array,
    ) -> jax.Array:
        """Accumulate direct token-topic counts by word.

        Args:
            p_ti: token-topic probabilities, shape ``(N, T)``.
            batch: token ids, shape ``(N,)``.

        Returns:
            Word-topic counts, shape ``(V, T)``.
        """
        return jnp.add.at(
            jnp.zeros((self.vocab_size, self.n_topics), dtype=p_ti.dtype),
            batch,
            p_ti,
            inplace=False,
        )

    @partial(jax.jit, static_argnums=0)
    def _calc_n_tw_big(
            self,
            *,
            p_ti: jax.Array,
            theta_ti: jax.Array,
            batch: jax.Array,
            ctx_bounds: jax.Array,
            weights_t: jax.Array | None = None,
    ) -> jax.Array:
        """Accumulate attention-adjusted word-topic counts by word.

        Args:
            p_ti: token-topic probabilities, shape ``(N, T)``.
            theta_ti: attention-smoothed token-topic probabilities, shape ``(N, T)``.
            batch: token ids, shape ``(N,)``.
            ctx_bounds: document boundary offsets, shape ``(D + 1,)``.
            weights_t: attention kernel, shape ``(K, T)``.

        Returns:
            Attention-adjusted word-topic counts, shape ``(V, T)``.
        """
        ratio = p_ti / jnp.maximum(theta_ti, self._eps)
        attn_ratio = self._apply_explicit_operator(
            x=ratio.T,
            ctx_bounds=ctx_bounds,
            weights_t=weights_t,
            transpose=True,
        ).T
        return jnp.add.at(
            jnp.zeros((self.vocab_size, self.n_topics), dtype=attn_ratio.dtype),
            batch,
            attn_ratio,
            inplace=False,
        )

    @partial(jax.jit, static_argnums=(0,), static_argnames=('grad_reg',))
    def _calc_phi(
            self,
            *,
            n_tw: jax.Array,
            N_tw: jax.Array,
            grad_reg: Callable,
            phi: jax.Array = None,
    ) -> jax.Array:
        """Update word-topic probabilities.

        Args:
            n_tw: direct word-topic counts, shape ``(V, T)``.
            N_tw: attention-adjusted word-topic counts, shape ``(V, T)``.
            grad_reg: regularization gradient callable over ``phi_base`` ``(V, T)``.
            phi: previous word-topic probabilities, shape ``(V, T)``; currently unused.

        Uses:
            ``self.n_w``: word prior/counts, shape ``(V,)``.
            ``self.n_t``: topic prior/counts, shape ``(T,)``.

        Returns:
            Row-normalized word-topic probabilities, shape ``(V, T)``.
        """
        del phi
        if self.n_w is None or self.n_t is None:
            raise ValueError('Word and topic priors must be initialized before phi update.')

        phi_base = n_tw / jnp.maximum(self.n_w[:, None], self._eps)
        reg_term = grad_reg(phi_base)
        phi_new = n_tw + phi_base * reg_term

        if self.use_big_n:
            p_w = self.n_w / jnp.maximum(jnp.sum(self.n_w), self._eps)
            p_t = self.n_t / jnp.maximum(jnp.sum(self.n_t), self._eps)
            subtract_term = phi_base * self.n_t[None, :] * (
                p_w[:, None] / jnp.maximum(p_t[None, :], self._eps)
            )
            phi_new = phi_new - subtract_term + phi_base * N_tw

        return self._norm(phi_new.T).T

    @partial(jax.jit, static_argnums=(0,), static_argnames=('grad_alpha_reg',))
    def _calc_weights_ti(
            self,
            *,
            weights_t: jax.Array,
            p_ti: jax.Array,
            theta_ti: jax.Array,
            phi: jax.Array,
            batch: jax.Array,
            ctx_bounds: jax.Array,
            grad_alpha_reg: Callable,
    ) -> jax.Array:
        """Update topic-specific attention weights.

        Args:
            weights_t: current attention kernel, shape ``(K, T)``.
            p_ti: token-topic probabilities, shape ``(N, T)``.
            theta_ti: attention-smoothed token-topic probabilities, shape ``(N, T)``.
            phi: word-topic probabilities, shape ``(V, T)``.
            batch: token ids, shape ``(N,)``.
            ctx_bounds: document boundary offsets, shape ``(D + 1,)``.
            grad_alpha_reg: regularization gradient callable over ``weights_t`` ``(K, T)``.

        Returns:
            Column-normalized attention kernel, shape ``(K, T)``.
        """
        ratio = p_ti / jnp.maximum(theta_ti, self._eps)
        phi_batch = phi[batch]
        doc_ids = self._get_doc_ids(batch=batch, ctx_bounds=ctx_bounds)

        kernel_len = weights_t.shape[0]
        center = kernel_len // 2
        seq_len = len(batch)

        center_term = jnp.sum(ratio * phi_batch, axis=0)
        past_terms = jnp.zeros((center, self.n_topics), dtype=weights_t.dtype)
        future_terms = jnp.zeros((center, self.n_topics), dtype=weights_t.dtype)

        for offset in range(1, center + 1):
            if seq_len <= offset:
                break

            same_doc = (doc_ids[offset:] == doc_ids[:-offset]).astype(weights_t.dtype)
            past_vals = jnp.sum(
                (ratio[offset:] * phi_batch[:-offset]) * same_doc[:, None],
                axis=0,
            )
            future_vals = jnp.sum(
                (ratio[:-offset] * phi_batch[offset:]) * same_doc[:, None],
                axis=0,
            )
            past_terms = past_terms.at[offset - 1].set(past_vals)
            future_terms = future_terms.at[offset - 1].set(future_vals)

        factors = jnp.concatenate(
            [
                past_terms[::-1],
                center_term[None, :].astype(weights_t.dtype),
                future_terms,
            ],
            axis=0,
        )
        reg_term = grad_alpha_reg(weights_t)
        updated = weights_t * (factors + reg_term)
        return self._norm(updated)

    @partial(jax.jit, static_argnums=0)
    def _get_doc_ids(
            self,
            *,
            batch: jax.Array,
            ctx_bounds: jax.Array,
    ) -> jax.Array:
        """Map each token position to a document id.

        Args:
            batch: token ids, shape ``(N,)``. Only its length is used.
            ctx_bounds: document boundary offsets, shape ``(D + 1,)``.

        Returns:
            Document id for each token position, shape ``(N,)``.
        """
        return jnp.searchsorted(ctx_bounds[1:], jnp.arange(len(batch)), side='right')
