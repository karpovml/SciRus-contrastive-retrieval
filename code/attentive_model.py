"""Backward-compatible import path for the attentive topic model.

The implementation lives in ``cartm.attentive``. Dimension comments for the
main model functions use:

- ``V``: vocabulary size;
- ``T``: number of topics;
- ``N``: number of tokens;
- ``D``: number of documents;
- ``K = 2 * ctx_len + 1``: attention kernel length.
"""

from .attentive import AttentiveTopicModel

__all__ = ['AttentiveTopicModel']
