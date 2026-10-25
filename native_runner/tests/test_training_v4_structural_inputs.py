"""Derived structural inputs (policy v4.2): data tables, feature maths, zero-start paths and checkpoint upgrade."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from native_runner.competitive_data import load_competitive_data
from native_runner.tests.test_training_v4_model import _ability_catalog, _batch, _catalog
from native_runner.training.v4 import components
from native_runner.training.v4.components import zero_start_parameters
from native_runner.training.v4 import model as model_module
from native_runner.training.v4.catalog import EntityArchetypeCatalogV1
from native_runner.training.v4.checkpoint import (
    checkpoint_contract,
    load_actor_critic_checkpoint,
    recent_action_checkpoint_contract,
    save_actor_critic_checkpoint,
    zero_start_parameter_names,
)
from native_runner.training.v4.config import CHILD_FEATURE_NAMES, ModelConfigV4
from native_runner.training.v4.mechanics import EffectSemanticCatalogV1, MechanicProfileCatalogV1
from native_runner.training.v4.model import UniversalCardPolicyV4
from native_runner.training.v4.structural_features import candidate_tower_features, child_impact_timing, tower_planes
from native_runner.training.v4.new_parameters import NewParameterScheduleV4
from native_runner.training.v4.structural_tables import (
    StructuralTablesV1,
    compiled_spell_facts,
    compiled_tower_hitpoints_level_11,
)

FIREBALL, ARROWS, ROCKET, LIGHTNING, ZAP, POISON = 28000000, 28000001, 28000003, 28000007, 28000008, 28000009
LOG, EARTHQUAKE, SNOWBALL, ROYAL_DELIVERY, VOID = 28000011, 28000014, 28000017, 28000018, 28000023
BARBARIAN_BARREL = 28000015
CHILD = {name: index for index, name in enumerate(CHILD_FEATURE_NAMES)}
PRINCESS_FULL = compiled_tower_hitpoints_level_11()[1][1]
REPOSITORY = Path(__file__).resolve().parents[2]


# --- Tables from the compiled game data ---------------------------------------------------------------------


def _card_row(card_id: int):
    return load_competitive_data("model_catalogs")["card_feature_rows"][str(card_id)]


@pytest.mark.parametrize("card_id", [FIREBALL, ROCKET, SNOWBALL, ZAP])
def test_single_hit_spells_match_the_compiled_card_rows(card_id: int) -> None:
    row = _card_row(card_id)
    expected = row["level_damage"] * row["crown_tower_damage_ratio"]
    assert compiled_spell_facts()[card_id].crown_tower_damage == pytest.approx(expected, abs=1.0)


def test_multi_wave_and_over_time_spells_add_up() -> None:
    facts = compiled_spell_facts()
    arrows = _card_row(ARROWS)
    assert facts[ARROWS].crown_tower_damage == pytest.approx(
        3 * arrows["level_damage"] * arrows["crown_tower_damage_ratio"], abs=1.0
    )
    poison = _card_row(POISON)
    assert facts[POISON].crown_tower_damage == pytest.approx(
        poison["level_damage_per_second"] * poison["effect_duration_ms"] / 1000 * poison["crown_tower_damage_ratio"],
        abs=1.5,
    )
    assert facts[EARTHQUAKE].crown_tower_damage == pytest.approx(81.92 * 3 * 0.6, abs=1.0)
    # Lightning strikes each target once: one hit on a tower.
    assert facts[LIGHTNING].crown_tower_damage == pytest.approx(413 * 2.56 * 0.25, abs=1.0)


def test_spells_missing_from_the_card_rows_are_resolved() -> None:
    facts = compiled_spell_facts()
    # The card rows leave the Log's damage empty; its rolling projectile carries it.
    assert _card_row(LOG)["level_damage"] is None
    assert facts[LOG].crown_tower_damage == pytest.approx(105 * 2.56 * 0.13, abs=0.5)
    # The Barbarian Barrel only rolls 4.5 tiles: cast from its own side it stops short of every enemy tower.
    assert facts[BARBARIAN_BARREL].crown_tower_damage == 0.0 and facts[BARBARIAN_BARREL].damage_known
    assert facts[BARBARIAN_BARREL].hits_ground and not facts[BARBARIAN_BARREL].hits_air
    # Royal Delivery's area ignores buildings, so it never reaches a tower.
    assert facts[ROYAL_DELIVERY].crown_tower_damage == 0.0 and facts[ROYAL_DELIVERY].damage_known
    # Scripted spells stay unknown rather than zero.
    assert not facts[VOID].damage_known and not facts[VOID].reach_known


def test_spell_reach_separates_ground_only_spells() -> None:
    facts = compiled_spell_facts()
    for ground_only in (LOG, EARTHQUAKE):
        assert facts[ground_only].hits_ground and not facts[ground_only].hits_air and facts[ground_only].reach_known
    for both in (FIREBALL, ZAP, ARROWS, POISON, LIGHTNING):
        assert facts[both].hits_air and facts[both].hits_ground


def test_tower_table_and_card_rows() -> None:
    table = compiled_tower_hitpoints_level_11()
    assert table[1] == table[2]
    assert table[0][0] == table[0][4] > table[1][1] > table[1][3] > table[1][2] > 0
    tables = StructuralTablesV1.for_card_ids((26000000, FIREBALL, LOG))
    assert tables.card_crown_tower_damage[2] == 0 and not tables.card_damage_known[2]
    assert tables.card_crown_tower_damage[3] == pytest.approx(compiled_spell_facts()[FIREBALL].crown_tower_damage)
    assert tables.card_spell_targets[4].tolist() == [0.0, 1.0, 1.0]


# --- Feature maths ------------------------------------------------------------------------------------------


def _alive_towers(batch, ratios):
    towers = batch.towers
    features = towers.features.clone()
    features[..., 0] = torch.tensor(ratios)
    features[..., 2] = 1.0
    features[..., 5] = torch.tensor(ratios) * 0.7
    # Fixture rows: own king, own left, own right, enemy king, enemy left, enemy right.
    return replace(towers, features=features, tower_type=torch.tensor([[0, 1, 2, 0, 1, 2]] * features.shape[0]))


def test_tower_planes_mark_standing_footprints_by_side() -> None:
    config = ModelConfigV4()
    towers = _alive_towers(_batch(), [1.0, 0.5, 1.0, 1.0, 0.25, 0.0])
    planes = tower_planes(towers, config)
    assert planes.shape == (2, 3, 32, 18)
    # Own left princess at (9, 2), extent ±0.5: only cell (x=8, y=1) has its centre (8.5, 1.5) inside.
    assert planes[0, 0, 1, 8] == pytest.approx(0.5)
    assert planes[0, 1, 1, 8] == 0
    # Enemy left princess at (9, 30) carries its ratio and hitpoints; the destroyed right one is absent.
    assert planes[0, 1, 29, 8] == pytest.approx(0.25)
    assert planes[0, 2, 29, 8] == pytest.approx(0.25 * 0.7)
    assert planes[0, 1, 28, 15] == 0
    assert int((planes[0, 1] > 0).sum()) == 2  # enemy king and enemy left princess, one cell each


def _tables(fraction_of_princess: float) -> StructuralTablesV1:
    catalog = _catalog()
    damage = torch.zeros(catalog.vocab_size)
    known = torch.zeros(catalog.vocab_size, dtype=torch.bool)
    damage[2] = fraction_of_princess * PRINCESS_FULL  # candidate 0's card
    known[2] = True
    known[3] = True  # candidate 1: a known card with no tower damage
    targets = torch.zeros(catalog.vocab_size, 3)
    targets[2] = torch.tensor([1.0, 1.0, 1.0])
    return StructuralTablesV1(
        card_crown_tower_damage=damage,
        card_damage_known=known,
        card_spell_targets=targets,
        tower_hitpoints_level_11=torch.tensor(compiled_tower_hitpoints_level_11()),
    )


def test_candidate_features_flag_the_tower_one_cast_kills() -> None:
    batch = _batch()
    towers = _alive_towers(batch, [0.1, 0.1, 0.1, 1.0, 0.25, 0.5])
    tables = _tables(0.3)
    features = candidate_tower_features(
        batch.candidates,
        towers,
        card_crown_tower_damage=tables.card_crown_tower_damage,
        card_damage_known=tables.card_damage_known,
        tower_hitpoints_level_11=tables.tower_hitpoints_level_11,
    )
    # 30% of a princess tower kills the enemy left one at 25% but not the right at 50%; own towers never count.
    assert features[0, 0].tolist() == pytest.approx([0.3 / 0.25 / 2, 1.0, 1.0, 0.0, 0.0, 1.0])
    assert features[0, 1].tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    # The Ability candidate and padding rows carry nothing, even when their card row has damage.
    assert not bool(features[0, 2].any()) and not bool(features[0, 4].any())
    assert not bool(features[1].any())


def test_candidate_features_reach_the_king_once_it_is_exposed() -> None:
    batch = _batch()
    towers = _alive_towers(batch, [1.0, 1.0, 1.0, 0.05, 0.0, 1.0])
    tables = _tables(0.3)
    features = candidate_tower_features(
        batch.candidates,
        towers,
        card_crown_tower_damage=tables.card_crown_tower_damage,
        card_damage_known=tables.card_damage_known,
        tower_hitpoints_level_11=tables.tower_hitpoints_level_11,
    )
    king_fraction = 0.3 * PRINCESS_FULL / compiled_tower_hitpoints_level_11()[0][0]
    assert features[0, 0, 4] == (1.0 if king_fraction >= 0.05 else 0.0)
    assert features[0, 0, 0] == pytest.approx(min(king_fraction / 0.05, 2.0) / 2)


def test_child_timing_measures_seconds_to_the_target_point() -> None:
    config = ModelConfigV4()
    groups = _batch().groups
    features = groups.child_features.clone()
    features[0, 0, CHILD["projectile_in_flight"]] = 1.0
    features[0, 0, CHILD["projectile_target_position_known"]] = 1.0
    features[0, 0, CHILD["projectile_target_x_over_board_width"]] = 4.8 / 18
    features[0, 0, CHILD["projectile_target_y_over_board_height"]] = 18.0 / 32
    features[0, 0, CHILD["velocity_y_over_10"]] = 0.025  # 0.25 tiles per tick
    # Child 1 is in flight but has not moved yet: no speed, unknown.
    features[0, 1, CHILD["projectile_in_flight"]] = 1.0
    features[0, 1, CHILD["projectile_target_position_known"]] = 1.0
    timing = child_impact_timing(replace(groups, child_features=features), config)
    # 10 tiles at 0.25 tiles per tick is 40 ticks, 2 seconds, 0.4 of the 5 s cap.
    assert timing[0, 0].tolist() == pytest.approx([0.4, 1.0])
    assert timing[0, 1].tolist() == [0.0, 0.0]
    assert not bool(timing[1].any())


# --- Zero start and checkpoint upgrade ----------------------------------------------------------------------


def _model(tables: StructuralTablesV1 | None = None) -> UniversalCardPolicyV4:
    catalog = _catalog()
    abilities = _ability_catalog(catalog)
    archetypes = EntityArchetypeCatalogV1(catalog.raw_card_ids, ("form:fixture_a", "form:fixture_b"))
    return UniversalCardPolicyV4(
        catalog,
        ability_catalog=abilities,
        entity_archetype_catalog=archetypes,
        effect_catalog=EffectSemanticCatalogV1.empty(catalog.raw_card_ids),
        mechanic_profile_catalog=MechanicProfileCatalogV1.empty(
            card_catalog=catalog, ability_catalog=abilities, entity_archetype_catalog=archetypes
        ),
        structural_tables=tables or _tables(0.3),
    )


def _rich_batch():
    """The fixture batch with standing towers, a lethal spell candidate and a projectile in flight."""

    batch = _batch()
    features = batch.groups.child_features.clone()
    features[:, 0, CHILD["projectile_in_flight"]] = 1.0
    features[:, 0, CHILD["projectile_target_position_known"]] = 1.0
    features[:, 0, CHILD["projectile_target_y_over_board_height"]] = 0.6
    features[:, 0, CHILD["velocity_y_over_10"]] = 0.02
    return replace(
        batch,
        towers=_alive_towers(batch, [1.0, 0.8, 1.0, 1.0, 0.25, 0.6]),
        groups=replace(batch.groups, child_features=features),
    )


def _structural_parameters(model: UniversalCardPolicyV4) -> dict[str, torch.Tensor]:
    return dict(zero_start_parameters(model))


def test_structural_paths_start_at_zero_and_change_nothing(monkeypatch) -> None:
    torch.manual_seed(5)
    model = _model().eval()
    assert set(_structural_parameters(model)) == {
        "card_encoder.spell_targets.weight",
        "child_encoder.timing.weight",
        "spatial_encoder.tower_input.weight",
        "candidate_encoder.tower_features.weight",
    }
    assert not any(bool(parameter.any()) for parameter in _structural_parameters(model).values())
    batch = _rich_batch()
    state = model.initial_state(batch.batch_size)
    with torch.no_grad():
        expected = model(batch, state)
        monkeypatch.setattr(components, "candidate_tower_features", lambda *a, **k: torch.rand(2, 5, 6))
        monkeypatch.setattr(components, "child_impact_timing", lambda *a, **k: torch.rand(2, 5, 2))
        monkeypatch.setattr(model_module, "tower_planes", lambda *a, **k: torch.rand(2, 3, 32, 18))
        model.card_encoder.catalog_spell_targets.uniform_()
        actual = model(batch, state)
    for name in ("value", "policy_context"):
        torch.testing.assert_close(getattr(actual, name), getattr(expected, name), rtol=0, atol=0)
    torch.testing.assert_close(actual.encoded.candidate_memory, expected.encoded.candidate_memory, rtol=0, atol=0)
    torch.testing.assert_close(actual.encoded.spatial_memory, expected.encoded.spatial_memory, rtol=0, atol=0)


def _recent_action_payload(model: UniversalCardPolicyV4, path: Path, *, optimizer=None) -> dict:
    """Write what a policy v4.1 checkpoint (recent actions, no structural inputs) contains."""

    save_actor_critic_checkpoint(path, model, optimizer=optimizer, training_stage="test", gamma_per_decision=0.99)
    payload = torch.load(path, weights_only=False)
    payload["model_state_dict"] = {
        name: value
        for name, value in payload["model_state_dict"].items()
        if name not in zero_start_parameter_names(model)
    }
    payload["contract"] = recent_action_checkpoint_contract(model)
    torch.save(payload, path)
    return payload


def test_v41_checkpoint_upgrade_is_output_identical_and_trainable(tmp_path) -> None:
    torch.manual_seed(6)
    source = _model().eval()
    with torch.no_grad():
        for name, parameter in source.named_parameters():
            if name not in zero_start_parameter_names(source):
                parameter.add_(0.01 * torch.randn_like(parameter))
    path = tmp_path / "v41.pt"
    _recent_action_payload(source, path)

    torch.manual_seed(7)
    upgraded = _model().eval()
    payload = load_actor_critic_checkpoint(path, upgraded, map_location="cpu")
    assert payload["contract"] == checkpoint_contract(upgraded)
    assert payload["upgraded_from_contract"] == recent_action_checkpoint_contract(upgraded)
    assert not any(bool(parameter.any()) for parameter in _structural_parameters(upgraded).values())

    batch = _rich_batch()
    state = source.initial_state(batch.batch_size)
    with torch.no_grad():
        expected = source(batch, state)
        actual = upgraded(batch, state)
    torch.testing.assert_close(actual.value, expected.value, rtol=0, atol=0)
    torch.testing.assert_close(actual.policy_context, expected.policy_context, rtol=0, atol=0)

    upgraded.train()
    context = upgraded(batch, state)
    (context.value.sum() + context.policy_context.square().sum()).backward()
    for name, parameter in _structural_parameters(upgraded).items():
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0, name


def test_v41_optimizer_moments_follow_names_and_new_parameters_start_empty(tmp_path) -> None:
    torch.manual_seed(8)
    source = _model()
    old_names = [name for name, _ in source.named_parameters() if name not in zero_start_parameter_names(source)]
    parameters = dict(source.named_parameters())
    optimizer = torch.optim.AdamW([parameters[name] for name in old_names], lr=1e-3)
    source(_rich_batch(), source.initial_state(2)).value.sum().backward()
    optimizer.step()
    path = tmp_path / "v41.pt"
    payload = _recent_action_payload(source, path)
    payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, path)

    target = _model()
    target_optimizer = torch.optim.AdamW(target.parameters(), lr=1e-3)
    load_actor_critic_checkpoint(path, target, optimizer=target_optimizer, map_location="cpu")
    names = {id(parameter): name for name, parameter in target.named_parameters()}
    restored = {names[id(parameter)]: state for parameter, state in target_optimizer.state.items()}
    original = optimizer.state_dict()["state"]
    assert set(restored) == {old_names[index] for index in original}
    assert not set(restored) & zero_start_parameter_names(target)
    for index, state in original.items():
        torch.testing.assert_close(restored[old_names[index]]["exp_avg"], state["exp_avg"])
    target(_rich_batch(), target.initial_state(2)).value.sum().backward()
    target_optimizer.step()


@pytest.mark.parametrize(
    "relative", ["checkpoints/General/checkpoint-step-00000460.pt", "checkpoints/2_6hog_expert/hog26-specialist2.pt"]
)
def test_repository_checkpoints_load_with_zero_structural_paths(relative: str) -> None:
    from native_runner.training.v4.factory import build_production_model_v4

    path = REPOSITORY / relative
    if not path.is_file() or path.stat().st_size < 1_000_000:
        pytest.skip("checkpoint file is not present")
    model = build_production_model_v4()
    payload = load_actor_critic_checkpoint(path, model, map_location="cpu")
    assert payload["contract"] == checkpoint_contract(model)
    assert "upgraded_from_contract" in payload
    assert not any(bool(parameter.any()) for parameter in _structural_parameters(model).values())
    fireball = model.catalog.vocab_id(FIREBALL)
    assert model.candidate_encoder.card_crown_tower_damage[fireball] == pytest.approx(172.2, abs=1.0)


# --- New-parameter learning-rate schedule ------------------------------------------------------------------


def _fixed_gradient_steps(model: UniversalCardPolicyV4, optimizer: torch.optim.Optimizer, steps: int) -> None:
    generator = torch.Generator().manual_seed(11)
    directions = [torch.randn(p.shape, generator=generator) for p in model.parameters()]
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = sum(((p * d) * (1.0 + step)).sum() + p.square().sum() for p, d in zip(model.parameters(), directions))
        loss.backward()
        optimizer.step()


def test_new_parameters_are_found_from_zero_start_layers() -> None:
    model = _model()
    assert [name for name, _ in zero_start_parameters(model)] == [
        "card_encoder.spell_targets.weight",
        "child_encoder.timing.weight",
        "spatial_encoder.tower_input.weight",
        "candidate_encoder.tower_features.weight",
    ]
    # The shared card encoder's projection is one tensor under every encoder that holds it.
    names = zero_start_parameter_names(model)
    assert "candidate_encoder.card_encoder.spell_targets.weight" in names
    assert {name for name in names if not name.endswith("card_encoder.spell_targets.weight")} == {
        "child_encoder.timing.weight",
        "spatial_encoder.tower_input.weight",
        "candidate_encoder.tower_features.weight",
    }


def test_scale_equals_a_larger_learning_rate_for_new_parameters_only() -> None:
    torch.manual_seed(5)
    scaled = _model()
    reference = _model()
    reference.load_state_dict(scaled.state_dict())
    learning_rate = 1e-3
    scaled_optimizer = torch.optim.AdamW(scaled.parameters(), lr=learning_rate, eps=1e-5, weight_decay=0.0)
    schedule = NewParameterScheduleV4(scaled, scale=30.0, decay_updates=0)
    assert len(schedule.attach(scaled_optimizer)) == 2
    new_ids = {id(p) for _name, p in zero_start_parameters(reference)}
    reference_optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in reference.parameters() if id(p) not in new_ids]},
            {"params": [p for p in reference.parameters() if id(p) in new_ids], "lr": 30 * learning_rate},
        ],
        lr=learning_rate,
        eps=1e-5,
        weight_decay=0.0,
    )
    _fixed_gradient_steps(scaled, scaled_optimizer, 3)
    _fixed_gradient_steps(reference, reference_optimizer, 3)
    expected = dict(reference.named_parameters())
    new_names = zero_start_parameter_names(scaled)
    for name, parameter in scaled.named_parameters():
        if name in new_names:
            assert parameter.abs().max() > 30 * learning_rate  # moved beyond what the base rate allows
            torch.testing.assert_close(parameter, expected[name], rtol=1e-5, atol=1e-7)
        else:
            torch.testing.assert_close(parameter, expected[name], rtol=0, atol=0)
    # Optimizer state keeps its single-group layout, so older checkpoints still load.
    assert len(scaled_optimizer.state_dict()["param_groups"]) == 1


def test_scale_fades_with_each_parameter_age_and_ages_travel_with_checkpoints(tmp_path) -> None:
    model = _model()
    schedule = NewParameterScheduleV4(
        model, scale=31.0, decay_updates=10, ages={"spatial_encoder.tower_input.weight": 5}
    )
    assert schedule.scale_for("child_encoder.timing.weight") == 31.0
    assert schedule.scale_for("spatial_encoder.tower_input.weight") == pytest.approx(16.0)
    for _ in range(5):
        schedule.advance()
    assert schedule.scale_for("child_encoder.timing.weight") == pytest.approx(16.0)
    assert schedule.scale_for("spatial_encoder.tower_input.weight") == 1.0
    metrics = schedule.metrics()
    assert metrics["new_rms/spatial_encoder.tower_input"] == 0.0
    assert metrics["new_age/spatial_encoder.tower_input"] == 10.0
    assert metrics["new_lr_scale/child_encoder.timing"] == pytest.approx(16.0)
    assert schedule.contract() == {"scale": 31.0, "decay_updates": 10}

    path = tmp_path / "aged.pt"
    save_actor_critic_checkpoint(
        path, model, training_stage="test", gamma_per_decision=0.99, parameter_ages=schedule.state()
    )
    restored = NewParameterScheduleV4(
        _model(), scale=31.0, decay_updates=10, ages=load_actor_critic_checkpoint(path, _model())["parameter_ages"]
    )
    assert restored.state() == schedule.state()
    # Without recorded ages (older checkpoints, imitation training) every new parameter counts as untrained.
    save_actor_critic_checkpoint(path, model, training_stage="test", gamma_per_decision=0.99)
    payload = load_actor_critic_checkpoint(path, _model())
    assert set(NewParameterScheduleV4(_model(), ages=payload.get("parameter_ages")).state().values()) == {0}


def test_default_schedule_adds_no_hooks_and_rejects_bad_settings() -> None:
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    schedule = NewParameterScheduleV4(model)
    assert schedule.attach(optimizer) == () and schedule.contract() is None
    with pytest.raises(ValueError):
        NewParameterScheduleV4(model, scale=0.0)
    with pytest.raises(ValueError):
        NewParameterScheduleV4(model, scale=2.0, decay_updates=-1)
