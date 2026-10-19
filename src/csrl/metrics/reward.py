"""Training rewards. They reuse the evaluation metrics so training and evaluation agree."""

from dataclasses import dataclass

from csrl.domain import constants
from csrl.metrics.code_switch import confusion_counts
from csrl.metrics.error_rates import mer_counts, rate
from csrl.metrics.hallucination import is_hallucination


@dataclass(frozen=True)
class RewardWeights:
    lambda_scr: float
    lambda_hal: float


@dataclass(frozen=True)
class RewardBreakdown:
    mer_term: float
    scr_term: float
    hallucination_term: float

    @property
    def total(self) -> float:
        return self.mer_term + self.scr_term + self.hallucination_term


def compute_reward(reference: str, hypothesis: str, weights: RewardWeights) -> RewardBreakdown:
    mer_value = rate(mer_counts(reference, hypothesis))
    confusions = confusion_counts(reference, hypothesis)
    return RewardBreakdown(
        mer_term=max(-mer_value, -constants.REWARD_MER_CLIP),
        scr_term=-weights.lambda_scr * (confusions.confusions_ne + confusions.confusions_en),
        hallucination_term=-weights.lambda_hal * float(is_hallucination(reference, hypothesis)),
    )
