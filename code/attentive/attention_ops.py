from functools import partial

import jax
import jax.numpy as jnp


class AttentionOperatorMixin:
    # optimize_weights просто что веса для обучения
    # _weights_update_locked это флаг для блокировки обновления весов
    def _should_optimize_weights(self) -> bool:
        return self.optimize_weights and not self._weights_update_locked

    @partial(jax.jit, static_argnums=0)
    def _clip_attention_weights(self, weights_t: jax.Array) -> jax.Array:
        """Clip and column-normalize attention weights.

        Args:
            weights_t: attention kernel, shape ``(K, T)``.

        Returns:
            Clipped normalized kernel, shape ``(K, T)``.
        """
        center = weights_t.shape[0] // 2
        clipped = weights_t
        if self.clip_context == 'center':
            clipped = clipped.at[center, :].set(0.0)
        elif self.clip_context == 'left':
            clipped = clipped.at[:center, :].set(0.0)
        elif self.clip_context == 'right':
            clipped = clipped.at[center + 1:, :].set(0.0)
        return self._norm(clipped)

    @partial(jax.jit, static_argnums=(0,), static_argnames=('transpose',))
    def _apply_explicit_operator(
            self,
            *,
            x: jax.Array,
            weights_t: jax.Array,
            ctx_bounds: jax.Array,
            transpose: bool = False,
    ) -> jax.Array:
        """Apply the explicit local attention operator.

        Args:
            x: topic-by-token matrix, shape ``(T, N)``.
            weights_t: attention kernel, shape ``(K, T)``.
            ctx_bounds: document boundary offsets, shape ``(D + 1,)``.
            transpose: if ``False`` applies the forward operator; if ``True`` swaps
                source/target slices and applies its transposed accumulation.

        Returns:
            Attention-smoothed matrix with the same shape as ``x``: ``(T, N)``.
        """
        if x.shape[1] == 0:
            return x

        center = weights_t.shape[0] // 2
        doc_ids = jnp.searchsorted(ctx_bounds[1:], jnp.arange(x.shape[1]), side='right')

        result = jnp.zeros_like(x)
        for shift in range(-center, center + 1):
            if abs(shift) >= x.shape[1]:
                continue

            weight = weights_t[center + shift]
            if shift < 0:
                source_slice = slice(None, shift)
                target_slice = slice(-shift, None)
                same_doc = doc_ids[-shift:] == doc_ids[:shift]
            elif shift > 0:
                source_slice = slice(shift, None)
                target_slice = slice(None, -shift)
                same_doc = doc_ids[:-shift] == doc_ids[shift:]
            else:
                source_slice = slice(None)
                target_slice = slice(None)
                same_doc = jnp.ones((x.shape[1],), dtype=bool)

            if transpose:
                source_slice, target_slice = target_slice, source_slice

            result = result.at[:, target_slice].add(
                x[:, source_slice] * (weight[:, None] * same_doc[None, :].astype(x.dtype))
            )

        return result
