from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp

from .. import metrics as mtc
from .. import regularization as reg


class RegularizationMetricMixin:
    """Registration, normalization, and metric helpers shared by the model."""

    def add_regularization(self, regularization: reg.Regularization):
        if not isinstance(regularization, reg.Regularization):
            regularization_name = getattr(regularization, '__name__', type(regularization).__name__)
            raise TypeError(
                f'Regularization [{regularization_name}] has to be a subclass of '
                f'the Regularization base class, got type {type(regularization)}'
            )

        self._regularizations[regularization.tag] = regularization

    def add_metric(self, metric: mtc.Metric):
        if not isinstance(metric, mtc.Metric):
            metric_name = getattr(metric, '__name__', type(metric).__name__)
            raise TypeError(
                f'Metric [{metric_name}] has to be a subclass of '
                f'the Metric base class, got type {type(metric)}'
            )

        self._metrics[metric.tag] = metric

    def add_alpha_regularization(self, *, tag: str, regularization: Callable[[jax.Array], jax.Array | float]):
        if not callable(regularization):
            raise TypeError(f'Alpha regularization [{tag}] must be callable.')
        self._alpha_regularizations[tag] = regularization

    def remove_regularization(self, tag: str):
        try:
            self._regularizations.pop(tag)
        except KeyError:
            print(
                f'Regularization with tag {tag} is not present. '
                f'Did you mean to use remove_metric?'
            )

    def remove_metric(self, tag: str):
        try:
            self._metrics.pop(tag)
        except KeyError:
            print(
                f'Metric with tag {tag} is not present. '
                f'Did you mean to use remove_regularization?'
            )

    def remove_alpha_regularization(self, tag: str):
        try:
            self._alpha_regularizations.pop(tag)
        except KeyError:
            print(f'Alpha regularization with tag {tag} is not present.')

    @partial(jax.jit, static_argnums=0)
    def _norm(self, x: jax.Array) -> jax.Array:
        """Column-normalize an array after clipping negatives to zero.

        Args:
            x: array with any leading dimension and normalized columns, commonly
                ``weights_t`` ``(K, T)`` or transposed ``phi`` ``(T, V)``.

        Returns:
            Array with the same shape as ``x``.
        """
        x = jnp.maximum(x, jnp.zeros_like(x))
        norm = x.sum(axis=0)
        return jnp.where(norm > self._eps, x / norm, jnp.zeros_like(x))

    @partial(jax.jit, static_argnums=0)
    def _norm_rows(self, x: jax.Array) -> jax.Array:
        """Row-normalize an array after clipping negatives to zero.

        Args:
            x: row-normalized matrix, commonly ``theta_ti`` with shape ``(N, T)``.

        Returns:
            Array with the same shape as ``x``.
        """
        x = jnp.maximum(x, jnp.zeros_like(x))
        norm = x.sum(axis=1, keepdims=True)
        return jnp.where(norm > self._eps, x / norm, jnp.zeros_like(x))

    def _compose_regularizations(self):
        regs = self._regularizations.values()
        reg_grad = jax.grad(lambda x: sum([0.0, ] + [reg(x) for reg in regs]))
        return jax.jit(reg_grad)

    def _compose_alpha_regularizations(self):
        regs = self._alpha_regularizations.values()
        reg_grad = jax.grad(lambda x: sum([0.0, ] + [reg(x) for reg in regs]))
        return jax.jit(reg_grad)

    def _calc_metrics(
            self,
            *,
            phi_it: jax.Array,
            phi_wt: jax.Array,
            theta: jax.Array,
            verbose: int,
    ):
        """Evaluate registered metrics.

        Args:
            phi_it: token-topic probabilities, shape ``(N, T)``.
            phi_wt: word-topic probabilities, shape ``(V, T)``.
            theta: attention-smoothed token-topic probabilities, shape ``(N, T)``.
            verbose: logging verbosity.
        """
        if len(self._metrics) == 0:
            return

        if verbose > 1:
            print('  Metrics:')
        for tag, metric in self._metrics.items():
            value = metric(phi_it=phi_it, phi_wt=phi_wt, theta=theta)
            if verbose > 1:
                print(f'    {tag}: {value:.04f}')
