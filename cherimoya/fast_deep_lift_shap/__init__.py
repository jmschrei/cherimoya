# fast_deep_lift_shap
# Author: Eugenio Mattei

"""
A fast DeepLIFT/DeepSHAP engine for the count head of a Cherimoya model.

It computes the attributions `cherimoya attribute` computes with tangermeme's
`deep_lift_shap` and Cherimoya's DeepLIFT rules (`attribution_ops`), for the
count head (`LogCountWrapper`), several times faster on a GPU. Each sequence
is forwarded once rather than once per reference, and the backward runs over
the sequence half of each sequence-reference pair only. The rules, the
references and the hypothetical projection are tangermeme's, and the CPU
tests hold the engine to tangermeme's `deep_lift_shap` to 1e-9 in float64.

`cherimoya attribute` runs it with ``"engine": "fast"``::

	from cherimoya import Cherimoya
	from cherimoya.fast_deep_lift_shap import Engine

	model = Cherimoya.load("model.torch", device="cuda", compile=False)
	engine = Engine(model, group=0, n_shuffles=20, random_state=0)

	mid = X.shape[-1] // 2
	result = engine.run(X, mid - 200, mid + 200)
	X_attr = result.attr                  # (n, 4, 400) float32

* `engine` holds the engine: the forward and backward passes, the rules, the
  precision modes and `Engine`.
* `references` draws the references, tangermeme's dinucleotide shuffles, in
  worker processes.
* `checks` holds the forward self-check and the audit against tangermeme's
  `deep_lift_shap`.
"""

from .engine import Engine
from .engine import RawResult
from .engine import Result
from .engine import VALIDATED_TANGERMEME_VERSIONS
from .engine import precision_mode
from .checks import AuditFailed
from .checks import audit
from .checks import audit_peaks
from .checks import forward_self_check
