from typing import Callable

import jax
import jax.numpy as jnp


class TrainingMixin:
    def _phase_count(self) -> int:
        base = 4 + self.n_attention_passes * 2
        if self.optimize_weights:
            base += 1
        return base

    @staticmethod
    def _advance_phase_progress(progress, *, phase: str):
        if progress is None:
            return
        progress.set_postfix(phase=phase)
        progress.update(1)

    def _step(
            self,
            *,
            batch: jax.Array,
            phi: jax.Array,
            grad_reg: Callable,
            grad_alpha_reg: Callable,
            ctx_bounds: jax.Array = None,
            weights_t: jax.Array | None = None,
            phase_progress=None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array | None]:
        """Run one EM-like attentive topic-model update.

        Args:
            batch: token ids, shape ``(N,)``.
            phi: current word-topic probabilities, shape ``(V, T)``.
            grad_reg: regularization gradient callable over ``phi`` ``(V, T)``.
            grad_alpha_reg: regularization gradient callable over ``weights_t`` ``(K, T)``.
            ctx_bounds: document boundary offsets, shape ``(D + 1,)``.
            weights_t: current attention kernel, shape ``(K, T)``.
            phase_progress: optional progress bar object.

        Returns:
            ``p_ti``: token-topic probabilities, shape ``(N, T)``.
            ``phi_new``: updated word-topic probabilities, shape ``(V, T)``.
            ``theta_ti``: attention-smoothed token-topic probabilities, shape ``(N, T)``.
            ``n_t_new``: topic counts, shape ``(T,)``.
            ``weights_new``: updated attention kernel, shape ``(K, T)``.
        """
        weights_t = self._clip_attention_weights(self._norm(weights_t.astype(jnp.float32)))
        p_ti = phi[batch]
        theta_ti = p_ti

        for _ in range(self.n_attention_passes):
            self._advance_phase_progress(phase_progress, phase='theta_refine')
            theta_ti = self._apply_explicit_operator(
                x=p_ti.T,
                ctx_bounds=ctx_bounds,
                weights_t=weights_t,
                transpose=False,
            ).T
            self._advance_phase_progress(phase_progress, phase='p_ti_refine')
            p_ti = self._update_p_ti(values=p_ti, theta_ti=theta_ti)

        self._advance_phase_progress(phase_progress, phase='n_tw')
        n_tw = self._calc_n_tw_small(p_ti=p_ti, batch=batch)
        self._advance_phase_progress(phase_progress, phase='N_tw')
        N_tw = self._calc_n_tw_big(
            p_ti=p_ti,
            theta_ti=theta_ti,
            batch=batch,
            ctx_bounds=ctx_bounds,
            weights_t=weights_t,
        )
        self._advance_phase_progress(phase_progress, phase='phi_update')
        phi_new = self._calc_phi(
            n_tw=n_tw,
            N_tw=N_tw,
            grad_reg=grad_reg,
            phi=phi,
        )
        self._advance_phase_progress(phase_progress, phase='n_t')
        n_t_new = jnp.sum(p_ti, axis=0)
        weights_new = weights_t
        if self._should_optimize_weights():
            self._advance_phase_progress(phase_progress, phase='alpha_update')
            weights_new = self._calc_weights_ti(
                weights_t=weights_t,
                p_ti=p_ti,
                theta_ti=theta_ti,
                phi=phi,
                batch=batch,
                ctx_bounds=ctx_bounds,
                grad_alpha_reg=grad_alpha_reg,
            )
            weights_new = self._clip_attention_weights(weights_new)
            if self.max_self_score is not None:
                reached_limit = bool(jnp.any(weights_new >= self.max_self_score))
                if reached_limit:
                    self._weights_update_locked = True
        return p_ti, phi_new, theta_ti, n_t_new, weights_new

    def fit(
            self,
            data: jax.Array,
            ctx_bounds: jax.Array,
            *,
            weights: jax.Array = None,
            max_iter: int = 1000,
            tol: float = 1e-3,
            verbose: int = 0,
            seed: int = 0,
            progress_bar: bool = False,
    ):
        """Fit the attentive topic model.

        Args:
            data: token ids, shape ``(N,)``.
            ctx_bounds: document boundary offsets, shape ``(D + 1,)``.
            weights: optional initial attention kernel, shape ``(K, T)``.
            max_iter: maximum number of model update iterations.
            tol: stopping tolerance on ``||phi_new - phi||``.
            verbose: logging verbosity.
            seed: random seed for initial ``phi``.
            progress_bar: whether to show tqdm progress bars.

        Sets:
            ``self.phi``: word-topic probabilities, shape ``(V, T)``.
            ``self.weights``: attention kernel, shape ``(K, T)``.
            ``self.n_t``: normalized topic prior, shape ``(T,)``.
            ``self.n_w``: word counts, shape ``(V,)``.
        """
        key = jax.random.key(seed)
        self._weights_update_locked = False
        self.phi = self._norm(jax.random.uniform(
            key=key,
            shape=(self.vocab_size, self.n_topics),
        ).T).T
        self._init_word_counts(data)
        self.n_t = jnp.full(
            shape=(self.n_topics, ),
            fill_value=1.0,
        )
        grad_regularization = self._compose_regularizations()
        grad_alpha_regularization = self._compose_alpha_regularizations()

        if weights is None:
            weights = self.weights if self.weights is not None else self._default_weights()
        weights_state = self._norm(weights.astype(jnp.float32))
        self.weights = weights_state

        iterator = range(max_iter)
        pbar = None
        if progress_bar:
            from tqdm.auto import tqdm
            pbar = tqdm(iterator, total=max_iter, desc=self.__class__.__name__, leave=True)
            iterator = pbar

        for it in iterator:
            phase_pbar = None
            if progress_bar:
                from tqdm.auto import tqdm
                phase_total = self._phase_count()
                phase_pbar = tqdm(
                    total=phase_total,
                    desc=f'{self.__class__.__name__} steps {it + 1}/{max_iter}',
                    leave=False,
                )
            phi_it, phi_new, theta, n_t_counts, weights_new = self._step(
                batch=data,
                phi=self.phi,
                grad_reg=grad_regularization,
                grad_alpha_reg=grad_alpha_regularization,
                ctx_bounds=ctx_bounds,
                weights_t=weights_state,
                phase_progress=phase_pbar,
            )
            weights_state = self._clip_attention_weights(weights_new)

            if phase_pbar is not None:
                phase_pbar.close()

            diff_norm = jnp.linalg.norm(phi_new - self.phi)
            if pbar is not None:
                pbar.set_postfix(diff_norm=f'{float(diff_norm):.4f}')
            if verbose > 0:
                print(f'Iteration [{it + 1}/{max_iter}], phi update diff norm: {diff_norm:.04f}')

            self._calc_metrics(
                phi_it=phi_it,
                phi_wt=phi_new,
                theta=theta,
                verbose=verbose,
            )

            self.phi = phi_new
            self.weights = weights_state
            self.n_t = n_t_counts / jnp.maximum(jnp.sum(n_t_counts), self._eps)
            if diff_norm < tol:
                break

        if pbar is not None:
            pbar.close()

    def calc_perplexity(
            self,
            data: jax.Array,
            ctx_bounds: jax.Array,
            weights: jax.Array = None,
            update_topic_prior: bool = False,
    ) -> float:
        """Calculate token-level perplexity for a token sequence.

        Args:
            data: token ids, shape ``(N,)``.
            ctx_bounds: document boundary offsets, shape ``(D + 1,)``.
            weights: attention kernel to use, shape ``(K, T)``. If omitted, callers
                should pass the fitted ``self.weights``.
            update_topic_prior: if true, recompute ``p_t`` from ``phi[data]``.

        Uses:
            ``self.phi``: word-topic probabilities, shape ``(V, T)``.
            ``self.n_t``: topic prior/counts, shape ``(T,)``.
            ``self.n_w``: word prior/counts, shape ``(V,)``.

        Returns:
            Scalar perplexity.
        """
        if self.phi is None or self.n_t is None:
            raise ValueError('Model must be fitted before perplexity calculation.')

        if self.n_w is None:
            self._init_word_counts(data)

        log_likelihood = 0.0
        p_ti = self.phi[data]
        theta_ti = p_ti

        theta_ti = self._apply_explicit_operator(
            x=p_ti.T,
            ctx_bounds=ctx_bounds,
            weights_t=weights,
            transpose=False,
        ).T

        p_t = None
        if update_topic_prior:
            p_t = jnp.sum(p_ti, axis=0)
            p_t = p_t / jnp.maximum(jnp.sum(p_t), self._eps)
        p_wi = self._calc_p_wi(phi=self.phi, theta_ti=theta_ti, batch=data, p_t=p_t)
        log_likelihood += float(jnp.sum(jnp.log(p_wi + self._eps)))
        return float(jnp.exp(-log_likelihood / len(data)))
