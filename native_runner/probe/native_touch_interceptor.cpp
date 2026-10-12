#include "native_touch_interceptor.h"

#include "native_touch_layout.h"

#include <android/log.h>
#include <jni.h>
#include <sys/mman.h>
#include <unistd.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include <pthread.h>

namespace {

constexpr const char *kLogTag = "CRNativeTouch";
constexpr std::uintptr_t kGameAppTouchOffset = 0x013d0be0;
constexpr std::uint8_t kGameAppTouchPrologue[16] = {
    0xfd, 0x7b, 0xbc, 0xa9, 0xf8, 0x5f, 0x01, 0xa9, 0xf6, 0x57, 0x02, 0xa9, 0xf4, 0x4f, 0x03, 0xa9,
};

constexpr std::int32_t kActionDown = 0;
constexpr std::int32_t kActionUp = 1;
constexpr std::int32_t kActionMove = 2;
constexpr std::int32_t kActionCancel = 3;

using GameAppTouch = void (*)(JNIEnv *, jclass, jint, jint, jint, jint);

enum class CaptureKind : std::int32_t {
  None = 0,
  Hand = 1,
  Arena = 2,
  // Tap on a live Champion/Hero carrier with no card selected.
  AbilityTap = 3,
  // A gesture that started on a hand the touch surface may not control.
  // Its tail is swallowed so the stock replay UI never sees half a gesture.
  Swallow = 4,
};

struct GestureCapture {
  CaptureKind kind = CaptureKind::None;
  std::int32_t pointer_id = -1;
  std::int32_t owner = -1;
  std::int32_t hand_index = -1;
};

NativeTouchBindings g_bindings;
GameAppTouch g_original_game_app_touch = nullptr;
pthread_mutex_t g_touch_mutex = PTHREAD_MUTEX_INITIALIZER;
GestureCapture g_capture;
NativeTouchSelectionSnapshot g_selection;
std::atomic<bool> g_installed{false};

// Runtime configuration, guarded by g_touch_mutex.  The defaults reproduce the
// original MuMu 1080x1920 behaviour exactly.
cr_native_touch::Layout g_layout = cr_native_touch::kNativeRender1080x1920;
bool g_custom_layout = false;
std::int32_t g_owner_filter = -1;
bool g_ability_tap = true;
bool g_calibrating = false;
std::uint64_t g_raw_sequence = 0;
NativeTouchRawSample g_last_down;
NativeTouchRawSample g_last_up;
std::uint64_t g_ability_taps_queued = 0;
std::uint64_t g_ability_taps_rejected = 0;

#define TOUCH_LOGI(...) __android_log_print(ANDROID_LOG_INFO, kLogTag, __VA_ARGS__)
#define TOUCH_LOGE(...) __android_log_print(ANDROID_LOG_ERROR, kLogTag, __VA_ARGS__)

bool native_render_ready() {
  return g_bindings.is_native_render_ready != nullptr && g_bindings.is_native_render_ready(g_bindings.context);
}

std::uint64_t native_render_generation() {
  return g_bindings.native_render_generation == nullptr ? 0 : g_bindings.native_render_generation(g_bindings.context);
}

void clear_capture_locked() { g_capture = {}; }

void clear_selection_locked() {
  if (g_selection.owner != -1 || g_selection.hand_index != -1) {
    g_selection.owner = -1;
    g_selection.hand_index = -1;
    ++g_selection.revision;
  }
}

void record_raw_sample_locked(std::int32_t action, std::int32_t x, std::int32_t y, std::int32_t pointer_id) {
  if (action != kActionDown && action != kActionUp) {
    return;
  }
  NativeTouchRawSample &sample = action == kActionDown ? g_last_down : g_last_up;
  sample.sequence = ++g_raw_sequence;
  sample.action = action;
  sample.x = x;
  sample.y = y;
  sample.pointer_id = pointer_id;
}

void forward_original(JNIEnv *env, jclass owner_class, jint action, jint x, jint y, jint pointer_id) {
  GameAppTouch original = g_original_game_app_touch;
  if (original != nullptr) {
    original(env, owner_class, action, x, y, pointer_id);
  }
}

extern "C" void native_touch_game_app_hook(JNIEnv *env, jclass owner_class, jint action, jint x, jint y,
                                           jint pointer_id) {
  const bool ready = native_render_ready();
  const std::uint64_t generation = ready ? native_render_generation() : 0;
  bool consume = false;
  bool selected = false;
  bool submit = false;
  bool ability_submit = false;
  std::int32_t submit_owner = -1;
  std::int32_t submit_hand_index = -1;
  std::int32_t selected_owner = -1;
  std::int32_t selected_hand_index = -1;
  cr_native_touch::WorldPoint submit_point;

  pthread_mutex_lock(&g_touch_mutex);
  record_raw_sample_locked(action, x, y, pointer_id);
  if (g_calibrating) {
    // Calibration only measures where the player taps.  Nothing is selected,
    // played, or forwarded to the stock UI while it is active.
    clear_capture_locked();
    pthread_mutex_unlock(&g_touch_mutex);
    return;
  }
  const cr_native_touch::Layout layout = g_layout;
  const std::int32_t owner_filter = g_owner_filter;
  const bool ability_tap = g_ability_tap && g_bindings.submit_ability_tap != nullptr;
  const auto owner_allowed = [owner_filter](std::int32_t owner) { return owner_filter < 0 || owner == owner_filter; };

  bool invalidated_captured_pointer = false;
  if (g_selection.generation != generation) {
    invalidated_captured_pointer = g_capture.kind != CaptureKind::None && g_capture.pointer_id == pointer_id;
    clear_capture_locked();
    clear_selection_locked();
    g_selection.generation = generation;
  }
  const bool captured_pointer = g_capture.kind != CaptureKind::None && g_capture.pointer_id == pointer_id;

  if (invalidated_captured_pointer) {
    // The matching DOWN belonged to the previous manager generation.
    consume = true;
  } else if (!ready) {
    // If native-render disappeared during a captured gesture, consume its
    // tail so the stock UI never receives an orphan UP/CANCEL.  Otherwise
    // clear stale selection and leave the event completely untouched.
    if (captured_pointer) {
      consume = true;
      if (action == kActionUp || action == kActionCancel) {
        clear_capture_locked();
      }
    } else {
      clear_capture_locked();
    }
    clear_selection_locked();
  } else if (action == kActionDown && g_capture.kind == CaptureKind::None) {
    const cr_native_touch::HandHit hand = cr_native_touch::hand_at_pixel(layout, x, y);
    const bool in_arena = cr_native_touch::arena_contains_pixel(layout, x, y);
    if (hand.hit) {
      g_capture = {
          owner_allowed(hand.owner) ? CaptureKind::Hand : CaptureKind::Swallow,
          pointer_id,
          hand.owner,
          hand.hand_index,
      };
      consume = true;
    } else if (g_selection.owner >= 0 && g_selection.hand_index >= 0 && in_arena) {
      g_capture = {
          CaptureKind::Arena,
          pointer_id,
          g_selection.owner,
          g_selection.hand_index,
      };
      consume = true;
    } else if (ability_tap && in_arena) {
      g_capture = {CaptureKind::AbilityTap, pointer_id, owner_filter, -1};
      consume = true;
    }
  } else if (captured_pointer) {
    consume = true;
    if (action == kActionUp) {
      if (g_capture.kind == CaptureKind::Hand) {
        const cr_native_touch::HandHit hand = cr_native_touch::hand_at_pixel(layout, x, y);
        if (hand.hit && hand.owner == g_capture.owner && hand.hand_index == g_capture.hand_index) {
          g_selection.owner = hand.owner;
          g_selection.hand_index = hand.hand_index;
          ++g_selection.revision;
          selected = true;
          selected_owner = hand.owner;
          selected_hand_index = hand.hand_index;
        } else if (!hand.hit && cr_native_touch::arena_contains_pixel(layout, x, y)) {
          // Dragging a card onto the arena plays it directly, like the
          // stock game.  Tap-card-then-tap-arena keeps working as before.
          submit = true;
          submit_owner = g_capture.owner;
          submit_hand_index = g_capture.hand_index;
          submit_point = cr_native_touch::arena_pixel_to_world(layout, x, y);
        }
      } else if (g_capture.kind == CaptureKind::Arena && g_selection.owner == g_capture.owner &&
                 g_selection.hand_index == g_capture.hand_index &&
                 cr_native_touch::arena_contains_pixel(layout, x, y)) {
        submit = true;
        submit_owner = g_capture.owner;
        submit_hand_index = g_capture.hand_index;
        submit_point = cr_native_touch::arena_pixel_to_world(layout, x, y);
      } else if (g_capture.kind == CaptureKind::AbilityTap && cr_native_touch::arena_contains_pixel(layout, x, y)) {
        ability_submit = true;
        submit_owner = g_capture.owner;
        submit_point = cr_native_touch::arena_pixel_to_world(layout, x, y);
      }
      clear_capture_locked();
    } else if (action == kActionCancel) {
      clear_capture_locked();
    } else if (action != kActionMove) {
      // Pointer DOWN/UP are normalized to 0/1 by the Java SurfaceView.
      // Unknown terminal-like actions are swallowed for this captured
      // pointer but do not mutate the selection.
    }
  }
  g_selection.gesture_captured = g_capture.kind != CaptureKind::None;
  const std::uint64_t selection_revision = g_selection.revision;
  pthread_mutex_unlock(&g_touch_mutex);

  if (selected) {
    TOUCH_LOGI("selected native-render hand owner=%d slot=%d revision=%llu", selected_owner, selected_hand_index,
               static_cast<unsigned long long>(selection_revision));
  }

  if (submit) {
    const bool queued = g_bindings.submit_hand_action != nullptr &&
                        g_bindings.submit_hand_action(g_bindings.context, submit_owner, submit_hand_index,
                                                      submit_point.x, submit_point.y);
    if (queued) {
      pthread_mutex_lock(&g_touch_mutex);
      if (g_selection.owner == submit_owner && g_selection.hand_index == submit_hand_index) {
        clear_selection_locked();
      }
      const std::uint64_t revision = g_selection.revision;
      pthread_mutex_unlock(&g_touch_mutex);
      TOUCH_LOGI("queued native-render placement owner=%d slot=%d world=(%d,%d) revision=%llu", submit_owner,
                 submit_hand_index, submit_point.x, submit_point.y, static_cast<unsigned long long>(revision));
    } else {
      TOUCH_LOGE("native-render placement queue rejected owner=%d slot=%d world=(%d,%d)", submit_owner,
                 submit_hand_index, submit_point.x, submit_point.y);
    }
  }

  if (ability_submit) {
    const bool queued = g_bindings.submit_ability_tap(g_bindings.context, submit_owner, submit_point.x, submit_point.y);
    pthread_mutex_lock(&g_touch_mutex);
    ++(queued ? g_ability_taps_queued : g_ability_taps_rejected);
    pthread_mutex_unlock(&g_touch_mutex);
    if (queued) {
      TOUCH_LOGI("queued native-render ability tap filter=%d world=(%d,%d)", submit_owner, submit_point.x,
                 submit_point.y);
    }
  }

  if (!consume) {
    forward_original(env, owner_class, action, x, y, pointer_id);
  }
}

void emit_absolute_jump(void *destination, const void *target) {
  // ldr x17, #8; br x17; .quad target
  const std::uint32_t instructions[2] = {0x58000051u, 0xd61f0220u};
  std::memcpy(destination, instructions, sizeof(instructions));
  std::memcpy(static_cast<std::uint8_t *>(destination) + 8, &target, sizeof(target));
}

bool make_page_writable(void *address, int protection) {
  const long page_size = sysconf(_SC_PAGESIZE);
  if (page_size <= 0) {
    return false;
  }
  const auto raw = reinterpret_cast<std::uintptr_t>(address);
  const auto page = raw & ~static_cast<std::uintptr_t>(page_size - 1);
  return mprotect(reinterpret_cast<void *>(page), static_cast<std::size_t>(page_size), protection) == 0;
}

} // namespace

bool install_native_touch_interceptor(std::uintptr_t libg_base, const NativeTouchBindings &bindings) {
#if !defined(__aarch64__)
  (void)libg_base;
  (void)bindings;
  return false;
#else
  if (libg_base == 0 || bindings.is_native_render_ready == nullptr || bindings.native_render_generation == nullptr ||
      bindings.submit_hand_action == nullptr) {
    TOUCH_LOGE("native touch bindings are incomplete");
    return false;
  }
  if (g_installed.load(std::memory_order_acquire)) {
    return true;
  }

  auto *target = reinterpret_cast<std::uint8_t *>(libg_base + kGameAppTouchOffset);
  if (std::memcmp(target, kGameAppTouchPrologue, sizeof(kGameAppTouchPrologue)) != 0) {
    TOUCH_LOGE("GameApp.nOnTouchEvent prologue mismatch at +0x%zx", static_cast<std::size_t>(kGameAppTouchOffset));
    return false;
  }

  void *trampoline = mmap(nullptr, 32, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (trampoline == MAP_FAILED) {
    TOUCH_LOGE("mmap touch trampoline failed");
    return false;
  }
  std::memcpy(trampoline, target, sizeof(kGameAppTouchPrologue));
  emit_absolute_jump(static_cast<std::uint8_t *>(trampoline) + 16, target + 16);
  __builtin___clear_cache(static_cast<char *>(trampoline), static_cast<char *>(trampoline) + 32);

  if (!make_page_writable(target, PROT_READ | PROT_WRITE | PROT_EXEC)) {
    TOUCH_LOGE("mprotect touch target RWX failed");
    munmap(trampoline, 32);
    return false;
  }
  g_bindings = bindings;
  g_original_game_app_touch = reinterpret_cast<GameAppTouch>(trampoline);
  emit_absolute_jump(target, reinterpret_cast<const void *>(&native_touch_game_app_hook));
  __builtin___clear_cache(reinterpret_cast<char *>(target), reinterpret_cast<char *>(target + 16));
  make_page_writable(target, PROT_READ | PROT_EXEC);
  g_installed.store(true, std::memory_order_release);
  TOUCH_LOGI("GameApp.nOnTouchEvent hook installed target=%p trampoline=%p default layout=1080x1920", target,
             trampoline);
  return true;
#endif
}

NativeTouchSelectionSnapshot native_touch_selection_snapshot() {
  pthread_mutex_lock(&g_touch_mutex);
  NativeTouchSelectionSnapshot snapshot = g_selection;
  snapshot.gesture_captured = g_capture.kind != CaptureKind::None;
  pthread_mutex_unlock(&g_touch_mutex);
  return snapshot;
}

void native_touch_clear_selection() {
  pthread_mutex_lock(&g_touch_mutex);
  clear_capture_locked();
  clear_selection_locked();
  g_selection.gesture_captured = false;
  pthread_mutex_unlock(&g_touch_mutex);
}

NativeTouchConfigSnapshot native_touch_config_snapshot() {
  NativeTouchConfigSnapshot snapshot;
  pthread_mutex_lock(&g_touch_mutex);
  snapshot.owner_filter = g_owner_filter;
  snapshot.ability_tap = g_ability_tap;
  snapshot.calibrating = g_calibrating;
  snapshot.custom_layout = g_custom_layout;
  const cr_native_touch::Layout &layout = g_layout;
  const std::int32_t values[18] = {
      layout.screen_width,      layout.screen_height,     layout.arena_left,        layout.arena_top,
      layout.arena_right,       layout.arena_bottom,      layout.world_left,        layout.world_top,
      layout.world_right,       layout.world_bottom,      layout.top_hand_y,        layout.bottom_hand_y,
      layout.hand_half_width,   layout.hand_half_height,  layout.hand_centers_x[0], layout.hand_centers_x[1],
      layout.hand_centers_x[2], layout.hand_centers_x[3],
  };
  std::memcpy(snapshot.layout_values, values, sizeof(values));
  snapshot.last_down = g_last_down;
  snapshot.last_up = g_last_up;
  snapshot.ability_taps_queued = g_ability_taps_queued;
  snapshot.ability_taps_rejected = g_ability_taps_rejected;
  pthread_mutex_unlock(&g_touch_mutex);
  return snapshot;
}

bool native_touch_set_layout(const std::int32_t (&values)[18]) {
  const cr_native_touch::Layout layout = {
      values[0],  values[1],  values[2],  values[3],  values[4],  values[5],
      values[6],  values[7],  values[8],  values[9],  values[10], values[11],
      values[12], values[13], {values[14], values[15], values[16], values[17]},
  };
  if (!cr_native_touch::layout_is_valid(layout)) {
    return false;
  }
  pthread_mutex_lock(&g_touch_mutex);
  g_layout = layout;
  g_custom_layout = true;
  clear_capture_locked();
  clear_selection_locked();
  g_selection.gesture_captured = false;
  pthread_mutex_unlock(&g_touch_mutex);
  TOUCH_LOGI("native touch layout set to %dx%d", layout.screen_width, layout.screen_height);
  return true;
}

void native_touch_reset_layout() {
  pthread_mutex_lock(&g_touch_mutex);
  g_layout = cr_native_touch::kNativeRender1080x1920;
  g_custom_layout = false;
  clear_capture_locked();
  clear_selection_locked();
  g_selection.gesture_captured = false;
  pthread_mutex_unlock(&g_touch_mutex);
}

bool native_touch_set_owner_filter(std::int32_t owner_filter) {
  if (owner_filter < -1 || owner_filter > 1) {
    return false;
  }
  pthread_mutex_lock(&g_touch_mutex);
  g_owner_filter = owner_filter;
  if (owner_filter >= 0 && g_selection.owner >= 0 && g_selection.owner != owner_filter) {
    clear_selection_locked();
  }
  pthread_mutex_unlock(&g_touch_mutex);
  return true;
}

void native_touch_set_ability_tap(bool enabled) {
  pthread_mutex_lock(&g_touch_mutex);
  g_ability_tap = enabled;
  pthread_mutex_unlock(&g_touch_mutex);
}

void native_touch_set_calibrating(bool enabled) {
  pthread_mutex_lock(&g_touch_mutex);
  g_calibrating = enabled;
  clear_capture_locked();
  clear_selection_locked();
  g_selection.gesture_captured = false;
  pthread_mutex_unlock(&g_touch_mutex);
}
