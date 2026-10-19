import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest
import soundfile
import torch

from csrl.application.checkpoint import load_checkpoint, save_checkpoint
from csrl.application.grpo import (
    GrpoObjectiveSettings,
    group_advantages,
    nonconstant_group_mask,
    policy_objective,
)
from csrl.application.policy import Policy
from csrl.application.script_gate import GateBiasSettings, ScriptGate, vocabulary_gate_classes
from csrl.application.settings import grpo_settings, sft_settings
from csrl.application.training import RunPaths, train_grpo, train_sft
from csrl.data.splits import split_by_speaker
from csrl.domain.entities import Utterance
from csrl.domain.enums import (
    AdvantageNormalization,
    DataSource,
    GateClass,
    ImportanceRatio,
    LossAggregation,
    Split,
)
from csrl.domain.errors import ConfigError, CsrlError, SplitError
from csrl.infrastructure.config import load_yaml
from tests.toy_backbone import EOS_ID, VOCABULARY, ToyBackbone

ROOT = Path(__file__).resolve().parent.parent


def _settings(**overrides: object) -> GrpoObjectiveSettings:
    base = GrpoObjectiveSettings(
        advantage_normalization=AdvantageNormalization.GROUP_STD,
        loss_aggregation=LossAggregation.TOKEN_MEAN,
        importance_ratio=ImportanceRatio.TOKEN,
        clip_low=0.2,
        clip_high=0.28,
        kl_coefficient=0.0,
        constant_normalizer=4,
    )
    return dataclasses.replace(base, **overrides)  # type: ignore[arg-type]


def test_group_advantages_std_and_mean() -> None:
    rewards = torch.tensor([1.0, 3.0, 2.0, 2.0])
    # group 1 mean 2, population std 1 -> [-1, 1]; group 2 constant -> [0, 0]
    std = group_advantages(rewards, 2, AdvantageNormalization.GROUP_STD)
    assert torch.allclose(std, torch.tensor([-1.0, 1.0, 0.0, 0.0]), atol=1e-5)
    mean = group_advantages(rewards, 2, AdvantageNormalization.GROUP_MEAN)
    assert torch.allclose(mean, torch.tensor([-1.0, 1.0, 0.0, 0.0]))
    scaled = group_advantages(rewards * 4, 2, AdvantageNormalization.GROUP_MEAN)
    assert torch.allclose(scaled, torch.tensor([-4.0, 4.0, 0.0, 0.0]))


def test_nonconstant_group_mask() -> None:
    mask = nonconstant_group_mask(torch.tensor([1.0, 3.0, 2.0, 2.0]), 2)
    assert mask.tolist() == [True, True, False, False]


def test_group_size_mismatch_raises() -> None:
    with pytest.raises(CsrlError):
        group_advantages(torch.zeros(5), 2, AdvantageNormalization.GROUP_STD)


def test_on_policy_loss_and_gradient() -> None:
    # With old == current, ratio is 1, so loss = -sum(A * mask) / tokens and
    # d loss / d logp_t = -A / tokens for each valid token.
    log_probs = torch.tensor([[-1.0, -2.0], [-0.5, -0.5]], requires_grad=True)
    mask = torch.tensor([[True, True], [True, False]])
    advantages = torch.tensor([1.0, -2.0])
    output = policy_objective(
        log_probs, log_probs.detach(), None, advantages, mask, _settings()
    )
    # tokens = 3; sum A*mask = 1 + 1 - 2 = 0
    assert float(output.loss.detach()) == pytest.approx(0.0)
    output.loss.backward()
    expected = torch.tensor([[-1 / 3, -1 / 3], [2 / 3, 0.0]])
    assert torch.allclose(log_probs.grad, expected)


def test_sequence_mean_aggregation() -> None:
    log_probs = torch.zeros(2, 2, requires_grad=True)
    mask = torch.tensor([[True, True], [True, False]])
    output = policy_objective(
        log_probs, log_probs.detach(), None, torch.tensor([1.0, 3.0]), mask,
        _settings(loss_aggregation=LossAggregation.SEQUENCE_MEAN),
    )
    # per sequence: -1 and -3, mean -2
    assert float(output.loss.detach()) == pytest.approx(-2.0)


def test_constant_normalizer_aggregation() -> None:
    log_probs = torch.zeros(2, 2, requires_grad=True)
    mask = torch.tensor([[True, True], [True, False]])
    output = policy_objective(
        log_probs, log_probs.detach(), None, torch.tensor([1.0, 3.0]), mask,
        _settings(loss_aggregation=LossAggregation.CONSTANT_NORMALIZER),
    )
    # sum = -(1+1+3) = -5, divided by 2 sequences * 4 = 8
    assert float(output.loss.detach()) == pytest.approx(-5 / 8)


def test_asymmetric_clip_token_ratio() -> None:
    # ratio exp(0.5) = 1.6487 for a positive advantage is clipped to 1 + clip_high = 1.28
    log_probs = torch.tensor([[0.5]], requires_grad=True)
    output = policy_objective(
        log_probs, torch.zeros(1, 1), None, torch.tensor([1.0]),
        torch.tensor([[True]]), _settings(),
    )
    assert float(output.loss.detach()) == pytest.approx(-1.28)
    assert output.clipped_fraction == pytest.approx(1.0)
    # negative advantage, ratio below 1 - clip_low: min picks the clipped value 0.8 * -1
    low = policy_objective(
        torch.tensor([[-1.0]], requires_grad=True), torch.zeros(1, 1), None,
        torch.tensor([-1.0]), torch.tensor([[True]]), _settings(),
    )
    assert float(low.loss.detach()) == pytest.approx(0.8)


def test_sequence_ratio_is_length_normalized() -> None:
    # token log ratios 0.2 and 0.4 -> sequence log ratio 0.3 -> ratio exp(0.3) = 1.3499,
    # clipped at 1.28 for positive advantage
    log_probs = torch.tensor([[0.2, 0.4]], requires_grad=True)
    output = policy_objective(
        log_probs, torch.zeros(1, 2), None, torch.tensor([1.0]),
        torch.tensor([[True, True]]),
        _settings(importance_ratio=ImportanceRatio.SEQUENCE,
                  loss_aggregation=LossAggregation.SEQUENCE_MEAN),
    )
    assert float(output.loss.detach()) == pytest.approx(-1.28)
    unclipped = policy_objective(
        torch.tensor([[0.05, 0.15]], requires_grad=True), torch.zeros(1, 2), None,
        torch.tensor([1.0]), torch.tensor([[True, True]]),
        _settings(importance_ratio=ImportanceRatio.SEQUENCE,
                  loss_aggregation=LossAggregation.SEQUENCE_MEAN),
    )
    assert float(unclipped.loss) == pytest.approx(-math.exp(0.1), rel=1e-5)


def test_kl_k3_zero_at_reference() -> None:
    log_probs = torch.tensor([[-1.0, -2.0]], requires_grad=True)
    output = policy_objective(
        log_probs, log_probs.detach(), log_probs.detach(), torch.tensor([0.0]),
        torch.tensor([[True, True]]), _settings(kl_coefficient=0.1),
    )
    assert output.mean_kl == pytest.approx(0.0)


def test_kl_requires_reference() -> None:
    with pytest.raises(CsrlError):
        policy_objective(
            torch.zeros(1, 1), torch.zeros(1, 1), None, torch.zeros(1),
            torch.tensor([[True]]), _settings(kl_coefficient=0.1),
        )


def test_gate_classes_and_bias() -> None:
    classes = vocabulary_gate_classes(VOCABULARY, special_ids={0, 1, 2})
    assert classes.tolist() == [
        GateClass.OTHER, GateClass.OTHER, GateClass.OTHER,
        GateClass.DEVANAGARI, GateClass.LATIN, GateClass.DEVANAGARI, GateClass.DEVANAGARI,
    ]
    gate = ScriptGate(4, classes)
    with torch.no_grad():
        gate.head.weight.zero_()
        gate.head.bias.copy_(torch.tensor([10.0, 0.0, 0.0]))  # confident Devanagari
    bias = gate.logit_bias(torch.zeros(1, 1, 4), GateBiasSettings(2.0, 0.9))
    # only the Latin token (id 4) is pushed down
    assert bias[0, 0].tolist() == [0.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0]
    weak = gate.logit_bias(torch.zeros(1, 1, 4), GateBiasSettings(2.0, 1.1))
    assert float(weak.abs().sum()) == 0.0


def _toy_policy(gate: bool = False) -> Policy:
    torch.manual_seed(0)
    backbone = ToyBackbone()
    script_gate = None
    if gate:
        script_gate = ScriptGate(
            backbone.hidden_size, vocabulary_gate_classes(VOCABULARY, {0, 1, 2})
        )
    return Policy(backbone, script_gate, None, max_new_tokens=6)


def test_greedy_matches_teacher_forced_argmax() -> None:
    policy = _toy_policy()
    audio = torch.randn(3, 1600)
    encoder = policy.backbone.encode(audio, torch.tensor([1600, 1600, 1600]))
    generation = policy.generate(encoder, temperature=1.0, greedy=True)
    log_probs = policy.token_log_probs(
        encoder, generation.content_ids, generation.action_mask, 1.0
    )
    # every greedy token must be the argmax under teacher forcing at its position
    prompt = torch.tensor([policy.backbone.prompt_token_ids()]).expand(3, -1)
    ids = torch.cat([prompt, generation.content_ids], dim=1)[:, :-1]
    mask = torch.cat([torch.ones_like(prompt, dtype=torch.bool), generation.action_mask], 1)[:, :-1]
    hidden = policy.backbone.decode_hidden(ids, mask, encoder)[:, prompt.shape[1] - 1 :]
    argmax = policy.backbone.project_logits(hidden).argmax(-1)
    valid = generation.action_mask
    assert torch.equal(argmax[valid], generation.content_ids[valid])
    assert bool((log_probs[~valid] == 0).all())


def test_generation_stops_after_eos() -> None:
    policy = _toy_policy()
    with torch.no_grad():
        policy.backbone.output.bias.fill_(-100.0)
        policy.backbone.output.bias[EOS_ID] = 100.0
    encoder = policy.backbone.encode(torch.randn(2, 1600), torch.tensor([1600, 1600]))
    generation = policy.generate(encoder, temperature=1.0, greedy=True)
    assert generation.content_ids.shape[1] == 1
    assert generation.texts == ["", ""]


def test_supervised_loss_decreases() -> None:
    policy = _toy_policy(gate=True)
    encoder = policy.backbone.encode(torch.randn(2, 1600), torch.tensor([1600, 1600]))
    references = ["मेरो phone राम्रो", "फोन"]
    parameters = list(policy.backbone.trainable_parameters()) + list(policy.gate.parameters())
    optimizer = torch.optim.Adam(parameters, lr=0.05)
    first = None
    for _ in range(60):
        loss = policy.supervised_loss(encoder, references)
        total = loss.token_loss + loss.gate_loss
        if first is None:
            first = float(total)
        optimizer.zero_grad()
        total.backward()
        optimizer.step()
    assert float(total) < 0.2 * first
    greedy = policy.generate(encoder, 1.0, greedy=True).texts
    assert greedy == references


def _utterance(index: int, speaker: str, seconds: float, text: str) -> Utterance:
    return Utterance(f"u{index}", f"u{index}.wav", seconds, speaker, text,
                     DataSource.CODESWITCH_INTERNAL)


def test_split_is_speaker_disjoint_and_deterministic() -> None:
    utterances = [_utterance(i, f"s{i % 10}", 1.0, "x") for i in range(100)]
    ratios = {Split.TRAIN: 0.7, Split.DEV: 0.1, Split.TEST: 0.2}
    first = split_by_speaker(utterances, ratios, seed=7)
    second = split_by_speaker(utterances, ratios, seed=7)
    assert first.speakers == second.speakers
    speaker_sets = [set(first.speakers[s]) for s in Split]
    assert all(not (a & b) for i, a in enumerate(speaker_sets) for b in speaker_sets[i + 1 :])
    assert sum(len(first.utterances[s]) for s in Split) == 100
    assert len(first.speakers[Split.TRAIN]) == 7


def test_split_rejects_bad_input() -> None:
    ratios = {Split.TRAIN: 0.7, Split.DEV: 0.1, Split.TEST: 0.2}
    with pytest.raises(SplitError):
        split_by_speaker([_utterance(0, "a", 1, "x"), _utterance(1, "b", 1, "x")], ratios, 0)
    with pytest.raises(SplitError):
        split_by_speaker([_utterance(0, "a", 1, "x")] * 3, ratios, 0)
    with pytest.raises(SplitError):
        split_by_speaker([], {Split.TRAIN: 1.0}, 0)


def test_shipped_configs_parse() -> None:
    sft_settings(load_yaml(ROOT / "configs" / "sft.yaml"))
    grpo = grpo_settings(load_yaml(ROOT / "configs" / "grpo.yaml"))
    assert grpo.objective.clip_high > grpo.objective.clip_low


def test_gspo_requires_sequence_mean() -> None:
    config = load_yaml(ROOT / "configs" / "grpo.yaml")
    config["objective"]["importance_ratio"] = "sequence"
    config["objective"]["loss_aggregation"] = "token_mean"
    with pytest.raises(ConfigError):
        grpo_settings(config)


def _write_audio_set(directory: Path, texts: list[str]) -> list[Utterance]:
    utterances = []
    for index, text in enumerate(texts):
        path = directory / f"u{index}.wav"
        soundfile.write(str(path), np.random.default_rng(index).standard_normal(1600) * 0.1,
                        16000)
        utterances.append(_utterance(index, f"s{index}", 0.1, text))
    return utterances


def test_end_to_end_sft_then_grpo(tmp_path: Path) -> None:
    texts = ["मेरो phone राम्रो", "फोन राम्रो", "मेरो फोन", "phone राम्रो"]
    data = _write_audio_set(tmp_path, texts)
    sft_config = load_yaml(ROOT / "configs" / "sft.yaml")
    sft_config.update(batch_size=2, max_steps=40, eval_every_steps=20, learning_rate=0.05,
                      max_new_tokens=6)
    sft_config["script_gate"]["enabled"] = True
    policy = _toy_policy(gate=True)
    sft_paths = RunPaths(tmp_path / "sft", tmp_path)
    best_sft = train_sft(policy, sft_settings(sft_config), data, data, sft_paths)
    assert sft_paths.best_checkpoint.is_file()
    assert (tmp_path / "sft" / "eval_dev.json").is_file()

    grpo_config = load_yaml(ROOT / "configs" / "grpo.yaml")
    grpo_config.update(batch_size=2, group_size=4, max_steps=6, eval_every_steps=3,
                       log_samples_every=3, learning_rate=0.01, max_new_tokens=6,
                       init_checkpoint=str(sft_paths.best_checkpoint), max_empty_fraction=1.0)
    grpo_config["objective"]["kl_coefficient"] = 0.05
    grpo_config["script_gate"]["enabled"] = True
    settings = grpo_settings(grpo_config)
    grpo_policy = _toy_policy(gate=True)
    load_checkpoint(sft_paths.best_checkpoint, grpo_policy.backbone, grpo_policy.gate)
    reference = Policy(grpo_policy.backbone.reference_copy(), None, None, max_new_tokens=6)
    best = train_grpo(grpo_policy, reference, settings, data, data,
                      RunPaths(tmp_path / "grpo", tmp_path))
    assert math.isfinite(best) and math.isfinite(best_sft)


def test_checkpoint_gate_mismatch_raises(tmp_path: Path) -> None:
    policy = _toy_policy(gate=True)
    path = tmp_path / "c.pt"
    save_checkpoint(path, policy.backbone, policy.gate, 1, 0.5)
    with pytest.raises(CsrlError):
        load_checkpoint(path, ToyBackbone(), None)
