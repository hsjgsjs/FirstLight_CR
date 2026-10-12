#pragma once

#include <cstdint>

struct NativeTouchBindings {
  // Both callbacks run on Android's touch-delivery thread.  The ready check
  // must be lock-free or short, and submit_hand_action must only enqueue the
  // action; it must not wait for a future render/game frame.
  bool (*is_native_render_ready)(void *context) = nullptr;
  // Must change whenever configure-native/reset publishes a new manager.
  // Selection is scoped to this token so it cannot leak into a later match.
  std::uint64_t (*native_render_generation)(void *context) = nullptr;
  bool (*submit_hand_action)(void *context, std::int32_t owner, std::int32_t hand_index, std::int32_t world_x,
                             std::int32_t world_y) = nullptr;
  // Optional.  Queues the ready Champion/Hero ability whose live carrier is
  // nearest to the tapped world point.  owner_filter is -1 (either owner) or
  // the one owner the touch surface may control.  Must only enqueue.
  bool (*submit_ability_tap)(void *context, std::int32_t owner_filter, std::int32_t world_x,
                             std::int32_t world_y) = nullptr;
  void *context = nullptr;
};

struct NativeStockAbilityReceipt {
  std::int32_t x = -1;
  std::int32_t y = -1;
  std::int32_t controller_slot = -1;
};

// Raw touch record kept for on-device layout calibration.
struct NativeTouchRawSample {
  std::uint64_t sequence = 0;
  std::int32_t action = -1;
  std::int32_t x = -1;
  std::int32_t y = -1;
  std::int32_t pointer_id = -1;
};

struct NativeTouchConfigSnapshot {
  // -1 lets taps control both owners (the original MuMu behaviour).
  std::int32_t owner_filter = -1;
  bool ability_tap = true;
  // While calibrating every touch is recorded and consumed, never acted on.
  bool calibrating = false;
  bool custom_layout = false;
  std::int32_t layout_values[18] = {};
  NativeTouchRawSample last_down;
  NativeTouchRawSample last_up;
  std::uint64_t ability_taps_queued = 0;
  std::uint64_t ability_taps_rejected = 0;
};

struct NativeTouchSelectionSnapshot {
  std::int32_t owner = -1;
  std::int32_t hand_index = -1;
  std::uint64_t revision = 0;
  std::uint64_t generation = 0;
  bool gesture_captured = false;
};

// Installs an inline hook at the v15.535.13 GameApp.nOnTouchEvent JNI entry.
// The caller must pass the already-validated base address of the exact libg
// build used by the native runner.
bool install_native_touch_interceptor(std::uintptr_t libg_base, const NativeTouchBindings &bindings);

NativeTouchSelectionSnapshot native_touch_selection_snapshot();
void native_touch_clear_selection();

NativeTouchConfigSnapshot native_touch_config_snapshot();
// values: screen W/H, arena L/T/R/B, world L/T/R/B, top/bottom hand Y,
// hand half width/height, four hand center X (cr_native_touch::Layout order).
bool native_touch_set_layout(const std::int32_t (&values)[18]);
void native_touch_reset_layout();
bool native_touch_set_owner_filter(std::int32_t owner_filter);
void native_touch_set_ability_tap(bool enabled);
void native_touch_set_calibrating(bool enabled);
