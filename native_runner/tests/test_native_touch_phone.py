"""Host checks for the phone-oriented native touch controls."""

from pathlib import Path
import shutil
import subprocess

import pytest

from native_runner.tests.cpp_source import cpp_function


PROBE = Path(__file__).resolve().parents[1] / "probe"


def _compile_and_run(tmp_path: Path, name: str, source: str) -> str:
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is required for native touch host tests")
    cpp = tmp_path / f"{name}.cpp"
    executable = tmp_path / f"{name}.exe"
    cpp.write_text(source, encoding="utf-8")
    subprocess.run(
        [compiler, "-std=c++17", "-O0", f"-I{PROBE}", str(cpp), "-o", str(executable)],
        check=True,
        capture_output=True,
        text=True,
    )
    return subprocess.run([str(executable)], check=True, capture_output=True, text=True).stdout


def test_touch_layout_geometry_and_runtime_validation(tmp_path) -> None:
    source = (PROBE / "test_native_touch_layout.cpp").read_text(encoding="utf-8")
    _compile_and_run(tmp_path, "touch_layout", source)


def test_ability_tap_picks_nearest_queueable_carrier_within_radius(tmp_path) -> None:
    glue = (PROBE / "native_touch_probe_glue.inc").read_text(encoding="utf-8")
    harness = r'''
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>

constexpr int kPlayerCount = 2;
constexpr int kMaxObservedObjects = 4;
constexpr std::int64_t kTouchAbilityTapRadius = 2500;
struct PlayerStateView { void* player = nullptr; };
struct NativeVectorView { void* data = nullptr; int count = 0; };
struct NativeEntityReferenceView {
    bool validated = true, has_value = true;
    int slot = -1;
    std::uint32_t native_object_id = 0;
    int owner = 0;
};
struct NativeChampionControllerView {
    int button_state = 2, remaining_cooldown_ms = 0, remaining_charges_raw = 1;
    int champion_count = 0;
    NativeEntityReferenceView champions[kMaxObservedObjects];
};
using Blob = std::array<unsigned char, 1024>;
Blob manager{}, roots[2]{};
std::array<unsigned char, 256> carriers[3]{};
void* objects[3] = {carriers[0].data(), carriers[1].data(), carriers[2].data()};
NativeChampionControllerView controllers[2];
bool controller_valid[2] = {true, true};
int world;
template<class T> T read_object_field(const void* p, std::size_t offset) {
    T value;
    std::memcpy(&value, static_cast<const unsigned char*>(p) + offset, sizeof(T));
    return value;
}
template<class T> void put(void* p, std::size_t offset, T value) {
    std::memcpy(static_cast<unsigned char*>(p) + offset, &value, sizeof(T));
}
bool process_range_is_readable(const void* p, std::size_t) { return p != nullptr; }
bool read_player_state(void* w, int owner, PlayerStateView* out) {
    if (w != &world || owner < 0 || owner > 1) return false;
    out->player = roots[owner].data();
    return true;
}
bool read_native_vector(void* p, std::size_t, int, NativeVectorView* out) {
    if (p != objects) return false;
    out->data = objects;
    out->count = 3;
    return true;
}
bool read_champion_controller(const void*, int, int slot, const void* p, void**, int,
                              NativeChampionControllerView* out) {
    if (slot < 1 || slot > 2 || p != &controllers[slot - 1] || !controller_valid[slot - 1]) return false;
    *out = controllers[slot - 1];
    return true;
}
bool ability_button_state_is_queueable(int state) { return state == 2 || state == 4; }
'''
    harness += cpp_function(glue, "nearest_ready_ability_carrier")
    harness += r'''
void place(int slot, int x, int y) {
    put(carriers[slot].data(), 0x7c, x);
    put(carriers[slot].data(), 0x80, y);
}
int main() {
    put(manager.data(), 0xa8, static_cast<void*>(&world));
    put(roots[1].data(), 0x10, static_cast<void*>(objects));
    for (int i = 0; i < 2; ++i) {
        put(roots[1].data(), 0x3a0 + i * sizeof(void*), static_cast<const void*>(&controllers[i]));
    }
    // Controller 1 has one carrier, controller 2 has two (e.g. two Hero cards).
    controllers[0].champion_count = 1;
    controllers[0].champions[0] = {true, true, 0, 501, 1};
    controllers[1].champion_count = 2;
    controllers[1].champions[0] = {true, true, 1, 502, 1};
    controllers[1].champions[1] = {true, true, 2, 503, 1};
    place(0, 3500, 20000);
    place(1, 9000, 20500);
    place(2, 14500, 21000);

    std::uint32_t id = 0;
    std::int64_t distance = 0;
    assert(nearest_ready_ability_carrier(manager.data(), 1, 9500, 20500, &id, &distance));
    assert(id == 502 && distance == 500 * 500);
    assert(nearest_ready_ability_carrier(manager.data(), 1, 14000, 22500, &id, &distance));
    assert(id == 503);
    // Outside the 2.5 tile radius of every carrier.
    assert(!nearest_ready_ability_carrier(manager.data(), 1, 9000, 27000, &id, &distance));
    // Cooldown, exhausted charges and a non-queueable state are never picked.
    controllers[1].remaining_cooldown_ms = 400;
    assert(nearest_ready_ability_carrier(manager.data(), 1, 6000, 20000, &id, &distance));
    assert(id == 501);
    assert(!nearest_ready_ability_carrier(manager.data(), 1, 9000, 20500, &id, &distance));
    controllers[1].remaining_cooldown_ms = 0;
    controllers[1].remaining_charges_raw = 0;
    assert(!nearest_ready_ability_carrier(manager.data(), 1, 9000, 20500, &id, &distance));
    controllers[1].remaining_charges_raw = 1;
    controllers[1].button_state = 9;  // NotEnoughElixir
    assert(!nearest_ready_ability_carrier(manager.data(), 1, 9000, 20500, &id, &distance));
    controllers[1].button_state = 2;
    // A controller that fails exact validation is skipped, not fatal.
    controller_valid[0] = false;
    assert(nearest_ready_ability_carrier(manager.data(), 1, 9000, 20500, &id, &distance));
    assert(!nearest_ready_ability_carrier(manager.data(), 1, 3500, 20000, &id, &distance));
    // Carriers must belong to the requested owner.
    assert(!nearest_ready_ability_carrier(manager.data(), 0, 9000, 20500, &id, &distance));
    std::puts("ability tap selection checks passed");
}
'''
    assert "ability tap selection checks passed" in _compile_and_run(tmp_path, "ability_tap", harness)


def test_probe_dispatches_touch_configuration_commands() -> None:
    probe = (PROBE / "cr_replay_probe.cpp").read_text(encoding="utf-8")
    glue = (PROBE / "native_touch_probe_glue.inc").read_text(encoding="utf-8")
    handler = cpp_function(glue, "handle_probe_native_touch_command")

    assert "handle_probe_native_touch_command(command, response, sizeof(response))" in probe
    for command in ('"touch layout default"', '"touch layout "', '"touch owner "', '"touch ability "',
                    '"touch calibrate "'):
        assert command in handler
    assert "bindings.submit_ability_tap = &probe_submit_native_touch_ability_tap;" in glue
    assert "resolve_ready_ability_request(manager, best_owner, kNativeObjectIdEntityKeyTag" in glue


def test_absent_hero_controller_no_longer_hides_runtime() -> None:
    probe = (PROBE / "cr_replay_probe.cpp").read_text(encoding="utf-8")
    absent = cpp_function(probe, "champion_controller_is_exact_absent")
    runtime = cpp_function(probe, "read_native_player_runtime")

    assert "read_object_field<std::int32_t>(selected_character, 0x40) < 0" in absent
    skip = runtime.index("if (champion_controller_is_exact_absent(owner_root, controller))")
    invalidate = runtime.index("result->ability_runtime_valid = false;", skip)
    assert skip < invalidate
