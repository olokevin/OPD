"""Prior-task (forgetting) evaluation run on the live ES vLLM engine.

arXiv:2601.20861 ("Evolutionary Strategies lead to Catastrophic Forgetting in LLMs")
measures forgetting by tracking a held-out prior-ability benchmark (HellaSwag) across
fine-tuning iterations.  Reproducing that needs the *curve*, not an endpoint, so the
eval has to run inside the ES loop.

Rather than re-implement benchmark preprocessing (and risk disagreeing with the
offline `lm_eval` numbers produced by `scripts/es/eval_forgetting.sh`), this wraps the
trainer's own vLLM engine as an `lm_eval` model.  Only `loglikelihood` is implemented,
which is all multiple-choice tasks use, so the scores come from lm-eval's own task
definitions and are identical to the offline harness.
"""

import os

import ray


def _lm_class():
    """Build the lm_eval adapter lazily -- `lm_eval` is not an ES dependency."""
    from lm_eval.api.model import LM

    class RayVLLMLM(LM):
        """`lm_eval` model backed by an ES Ray engine actor."""

        def __init__(self, engine, tokenizer, batch_size=512, max_len=4096):
            super().__init__()
            self.engine = engine
            self.tokenizer = tokenizer
            self.batch_size = batch_size
            self.max_len = max_len
            self._eot = tokenizer.eos_token_id or tokenizer.bos_token_id or 0

        def loglikelihood(self, requests, disable_tqdm=False):
            from vllm import SamplingParams, TokensPrompt

            enc = self.tokenizer
            prompts, spans = [], []
            for req in requests:
                ctx, cont = req.args
                if ctx:
                    ctx_ids = enc(ctx, add_special_tokens=False)["input_ids"]
                    ids = enc(ctx + cont, add_special_tokens=False)["input_ids"]
                else:
                    ctx_ids = [self._eot]
                    ids = ctx_ids + enc(cont, add_special_tokens=False)["input_ids"]
                n_cont = max(1, len(ids) - len(ctx_ids))
                ids = ids[-self.max_len:]
                prompts.append(TokensPrompt(prompt_token_ids=ids))
                spans.append(min(n_cont, len(ids) - 1))

            sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
            out = []
            for b in range(0, len(prompts), self.batch_size):
                res = ray.get(self.engine.generate.remote(
                    prompts[b : b + self.batch_size], sp, use_tqdm=False))
                for r, n in zip(res, spans[b : b + self.batch_size]):
                    lps = r.prompt_logprobs  # [None, {tok: Logprob}, ...], token-aligned
                    tot = 0.0
                    for pos in range(len(lps) - n, len(lps)):
                        d = lps[pos]
                        if d:
                            tot += float(next(iter(d.values())).logprob)
                    out.append((tot, False))
            return out

        def loglikelihood_rolling(self, requests, disable_tqdm=False):
            raise NotImplementedError("prior-task eval uses multiple-choice tasks only")

        def generate_until(self, requests, disable_tqdm=False):
            raise NotImplementedError("prior-task eval uses multiple-choice tasks only")

    return RayVLLMLM


def evaluate_prior_tasks(engine, tokenizer, tasks, limit=None, batch_size=512,
                         max_len=4096, num_fewshot=0):
    """-> {"forget/<task>": accuracy in %}.  `limit` subsamples each task."""
    import lm_eval

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    lm = _lm_class()(engine, tokenizer, batch_size=batch_size, max_len=max_len)
    res = lm_eval.simple_evaluate(
        model=lm, tasks=list(tasks), limit=limit, num_fewshot=num_fewshot,
        bootstrap_iters=0, verbosity="ERROR",
    )["results"]

    out = {}
    for task, m in res.items():
        # `acc_norm` where the task defines it (HellaSwag/ARC/OBQA), else `acc`.
        key = "acc_norm,none" if "acc_norm,none" in m else "acc,none"
        if key in m:
            out[f"forget/{task}"] = 100.0 * float(m[key])
    if out:
        out["forget/mean"] = sum(out.values()) / len(out)
    return out
