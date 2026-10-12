// Integration shim for cr_replay_probe.cpp.
//
// Include native_touch_interceptor.h with the other local headers, then
// include this file immediately after native_replay_runner_hook() and before
// install_inline_hook().  At that location every GameState/observation helper
// used below has already been defined in the probe's anonymous namespace.

bool probe_native_touch_ready(void *) {
  if (g_runner_mode.load(std::memory_order_acquire) != static_cast<std::int32_t>(RunnerMode::NativeRender)) {
    return false;
  }
  const std::uint64_t submitted = g_native_render_submitted_sequence.load(std::memory_order_acquire);
  const std::uint64_t loaded = g_native_render_loaded_sequence.load(std::memory_order_acquire);
  void *const manager = g_native_render_manager.load(std::memory_order_acquire);
  if (submitted == 0 || loaded != submitted || manager == nullptr ||
      g_native_render_tick.load(std::memory_order_acquire) < 0) {
    return false;
  }

  // A finalized ReplayBattleController still owns a readable manager while
  // its result screen is visible.  Do not steal result-screen taps merely
  // because nativeRenderReady remains true at the transport level.
  BattleResultView result;
  const bool readable = read_battle_result(read_object_field<void *>(manager, 0xa8), &result);
  return !readable || !result.finalized;
}

std::uint64_t probe_native_touch_generation(void *) {
  return g_native_render_loaded_sequence.load(std::memory_order_acquire);
}

bool probe_submit_native_touch_hand_action(void *, std::int32_t owner, std::int32_t hand_index, std::int32_t world_x,
                                           std::int32_t world_y) {
  constexpr std::int32_t kTouchCommandDelayTicks = 2;
  char command_json[768] = {};

  pthread_mutex_lock(&g_control_mutex);
  void *const manager = g_native_render_manager.load(std::memory_order_acquire);
  const std::uint64_t submitted = g_native_render_submitted_sequence.load(std::memory_order_acquire);
  const bool ready =
      g_runner_mode.load(std::memory_order_acquire) == static_cast<std::int32_t>(RunnerMode::NativeRender) &&
      submitted != 0 && g_native_render_loaded_sequence.load(std::memory_order_acquire) == submitted &&
      manager != nullptr;
  const std::int32_t current_tick = g_native_render_tick.load(std::memory_order_acquire);

  // Never wait on the Android touch thread.  If a host action currently
  // occupies the one-command mailbox, reject this click and leave its card
  // selected so the user can tap the arena again.
  const bool mailbox_available = g_pending_native_command_sequence == g_processed_native_command_sequence;
  DeployRequest request;
  const bool resolved = ready && mailbox_available && current_tick >= 0 &&
                        current_tick <= INT32_MAX - kTouchCommandDelayTicks &&
                        resolve_hand_deploy_request(manager, owner, hand_index, world_x, world_y, &request);
  const bool formatted = resolved && format_deploy_command(request, current_tick + kTouchCommandDelayTicks,
                                                           command_json, sizeof(command_json));

  if (formatted) {
    ++g_pending_native_command_sequence;
    std::memcpy(g_pending_native_command_json, command_json, std::strlen(command_json) + 1);
    g_last_native_injection_succeeded = false;
    pthread_cond_broadcast(&g_control_cond);
  }
  pthread_mutex_unlock(&g_control_mutex);
  return formatted;
}

// A tap selects a carrier whose ground position is within this many world
// units (2.5 tiles).  Rendered sprites stand above their ground point, so the
// radius is deliberately forgiving; the nearest eligible carrier wins.
constexpr std::int64_t kTouchAbilityTapRadius = 2500;

// Finds the owner's queueable Champion/Hero carrier nearest to a world point.
// Every carrier still passes the exact controller validation in
// read_champion_controller; the final command is re-resolved by its native id.
bool nearest_ready_ability_carrier(void *manager, std::int32_t owner, std::int32_t world_x, std::int32_t world_y,
                                   std::uint32_t *native_object_id, std::int64_t *distance_squared) {
  if (manager == nullptr || native_object_id == nullptr || distance_squared == nullptr || owner < 0 ||
      owner >= kPlayerCount) {
    return false;
  }
  void *const world = read_object_field<void *>(manager, 0xa8);
  PlayerStateView player;
  if (!read_player_state(world, owner, &player)) {
    return false;
  }
  void *const object_manager = read_object_field<void *>(player.player, 0x10);
  NativeVectorView object_vector;
  if (!read_native_vector(object_manager, 0x08, 100000, &object_vector) ||
      (object_vector.count != 0 &&
       !process_range_is_readable(object_vector.data,
                                  static_cast<std::size_t>(object_vector.count) * sizeof(void *)))) {
    return false;
  }
  auto **const objects = static_cast<void **>(object_vector.data);
  bool found = false;
  for (std::int32_t controller_index = 0; controller_index < 2; ++controller_index) {
    const void *const controller = read_object_field<const void *>(
        player.player, 0x3a0 + static_cast<std::size_t>(controller_index) * sizeof(void *));
    if (controller == nullptr) {
      continue;
    }
    NativeChampionControllerView ability;
    if (!read_champion_controller(player.player, owner, controller_index + 1, controller, objects,
                                  object_vector.count, &ability) ||
        !ability_button_state_is_queueable(ability.button_state) || ability.remaining_cooldown_ms != 0 ||
        ability.remaining_charges_raw == 0) {
      continue;
    }
    for (std::int32_t index = 0; index < ability.champion_count; ++index) {
      const NativeEntityReferenceView &candidate = ability.champions[index];
      if (!candidate.validated || !candidate.has_value || candidate.owner != owner || candidate.slot < 0 ||
          candidate.slot >= object_vector.count || !process_range_is_readable(objects[candidate.slot], 0x84)) {
        continue;
      }
      const std::int64_t dx =
          static_cast<std::int64_t>(read_object_field<std::int32_t>(objects[candidate.slot], 0x7c)) - world_x;
      const std::int64_t dy =
          static_cast<std::int64_t>(read_object_field<std::int32_t>(objects[candidate.slot], 0x80)) - world_y;
      const std::int64_t distance = dx * dx + dy * dy;
      if (distance <= kTouchAbilityTapRadius * kTouchAbilityTapRadius && (!found || distance < *distance_squared)) {
        found = true;
        *distance_squared = distance;
        *native_object_id = candidate.native_object_id;
      }
    }
  }
  return found;
}

bool probe_submit_native_touch_ability_tap(void *, std::int32_t owner_filter, std::int32_t world_x,
                                           std::int32_t world_y) {
  // t = current tick keeps one tick of slack for the mailbox to be drained
  // by the next stock step (format_ability_command writes t2 = t + 1).
  constexpr std::int32_t kTouchAbilityDelayTicks = 2;
  char command_json[384] = {};

  pthread_mutex_lock(&g_control_mutex);
  void *const manager = g_native_render_manager.load(std::memory_order_acquire);
  const std::uint64_t submitted = g_native_render_submitted_sequence.load(std::memory_order_acquire);
  const bool ready =
      g_runner_mode.load(std::memory_order_acquire) == static_cast<std::int32_t>(RunnerMode::NativeRender) &&
      submitted != 0 && g_native_render_loaded_sequence.load(std::memory_order_acquire) == submitted &&
      manager != nullptr;
  const std::int32_t current_tick = g_native_render_tick.load(std::memory_order_acquire);
  // Never wait on the Android touch thread (see the hand-action path).
  const bool mailbox_available = g_pending_native_command_sequence == g_processed_native_command_sequence;

  bool found = false;
  std::int32_t best_owner = -1;
  std::uint32_t best_id = 0;
  std::int64_t best_distance = 0;
  if (ready && mailbox_available && current_tick >= 0 && current_tick <= INT32_MAX - kTouchAbilityDelayTicks) {
    for (std::int32_t owner = 0; owner < kPlayerCount; ++owner) {
      if (owner_filter >= 0 && owner != owner_filter) {
        continue;
      }
      std::uint32_t id = 0;
      std::int64_t distance = 0;
      if (nearest_ready_ability_carrier(manager, owner, world_x, world_y, &id, &distance) &&
          (!found || distance < best_distance)) {
        found = true;
        best_owner = owner;
        best_id = id;
        best_distance = distance;
      }
    }
  }
  AbilityRequest request;
  AbilityResolutionStatus status = AbilityResolutionStatus::Unavailable;
  const bool resolved = found && best_id <= static_cast<std::uint32_t>(INT32_MAX) &&
                        resolve_ready_ability_request(manager, best_owner, kNativeObjectIdEntityKeyTag,
                                                      static_cast<std::int32_t>(best_id), &request, nullptr, &status) &&
                        status == AbilityResolutionStatus::Resolved;
  const bool formatted = resolved && format_ability_command(request, current_tick + kTouchAbilityDelayTicks - 2,
                                                            command_json, sizeof(command_json));
  if (formatted) {
    ++g_pending_native_command_sequence;
    std::memcpy(g_pending_native_command_json, command_json, std::strlen(command_json) + 1);
    g_last_native_injection_succeeded = false;
    pthread_cond_broadcast(&g_control_cond);
  }
  pthread_mutex_unlock(&g_control_mutex);
  return formatted;
}

bool install_probe_native_touch(std::uintptr_t libg_base) {
  NativeTouchBindings bindings;
  bindings.is_native_render_ready = &probe_native_touch_ready;
  bindings.native_render_generation = &probe_native_touch_generation;
  bindings.submit_hand_action = &probe_submit_native_touch_hand_action;
  bindings.submit_ability_tap = &probe_submit_native_touch_ability_tap;
  return install_native_touch_interceptor(libg_base, bindings);
}

bool format_probe_native_touch_status(char *response, std::size_t response_size) {
  if (response == nullptr || response_size == 0) {
    return false;
  }
  const NativeTouchSelectionSnapshot selection = native_touch_selection_snapshot();
  const NativeTouchConfigSnapshot config = native_touch_config_snapshot();
  const std::int32_t *const l = config.layout_values;
  const int written = std::snprintf(
      response, response_size,
      "{\"ok\":true,\"mode\":\"native-render-touch\","
      "\"selectedOwner\":%d,\"selectedHandIndex\":%d,"
      "\"selectionRevision\":%llu,\"generation\":%llu,"
      "\"gestureCaptured\":%s,"
      "\"ownerFilter\":%d,\"abilityTap\":%s,\"calibrating\":%s,\"customLayout\":%s,"
      "\"layout\":[%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d],"
      "\"lastDown\":{\"sequence\":%llu,\"x\":%d,\"y\":%d,\"pointer\":%d},"
      "\"lastUp\":{\"sequence\":%llu,\"x\":%d,\"y\":%d,\"pointer\":%d},"
      "\"abilityTapsQueued\":%llu,\"abilityTapsRejected\":%llu}",
      selection.owner, selection.hand_index, static_cast<unsigned long long>(selection.revision),
      static_cast<unsigned long long>(selection.generation), selection.gesture_captured ? "true" : "false",
      config.owner_filter, config.ability_tap ? "true" : "false", config.calibrating ? "true" : "false",
      config.custom_layout ? "true" : "false", l[0], l[1], l[2], l[3], l[4], l[5], l[6], l[7], l[8], l[9], l[10],
      l[11], l[12], l[13], l[14], l[15], l[16], l[17], static_cast<unsigned long long>(config.last_down.sequence),
      config.last_down.x, config.last_down.y, config.last_down.pointer_id,
      static_cast<unsigned long long>(config.last_up.sequence), config.last_up.x, config.last_up.y,
      config.last_up.pointer_id, static_cast<unsigned long long>(config.ability_taps_queued),
      static_cast<unsigned long long>(config.ability_taps_rejected));
  return written > 0 && static_cast<std::size_t>(written) < response_size;
}

// "touch layout default", "touch layout <18 ints>", "touch owner -1|0|1",
// "touch ability on|off", "touch calibrate on|off".  Returns false when the
// command is not a touch configuration command at all.
bool handle_probe_native_touch_command(const char *command, char *response, std::size_t response_size) {
  const auto ok_status = [&]() { return format_probe_native_touch_status(response, response_size); };
  const auto error = [&](const char *message) {
    std::snprintf(response, response_size, "{\"ok\":false,\"error\":\"%s\"}", message);
    return true;
  };
  if (std::strcmp(command, "touch layout default") == 0) {
    native_touch_reset_layout();
    ok_status();
    return true;
  }
  if (std::strncmp(command, "touch layout ", 13) == 0) {
    std::int32_t values[18] = {};
    const char *cursor = command + 13;
    for (std::int32_t index = 0; index < 18; ++index) {
      int consumed = 0;
      if (std::sscanf(cursor, "%d%n", &values[index], &consumed) != 1 || consumed <= 0) {
        return error("touch layout expects 18 integers or 'default'");
      }
      cursor += consumed;
    }
    while (*cursor == ' ' || *cursor == '\t') {
      ++cursor;
    }
    if (*cursor != '\0') {
      return error("unexpected trailing touch layout fields");
    }
    if (!native_touch_set_layout(values)) {
      return error("touch layout geometry is invalid");
    }
    ok_status();
    return true;
  }
  if (std::strncmp(command, "touch owner ", 12) == 0) {
    std::int32_t owner = 0;
    int consumed = 0;
    if (std::sscanf(command + 12, "%d%n", &owner, &consumed) != 1 || command[12 + consumed] != '\0' ||
        !native_touch_set_owner_filter(owner)) {
      return error("touch owner expects -1, 0 or 1");
    }
    ok_status();
    return true;
  }
  const auto toggle = [&](const char *prefix, void (*setter)(bool)) {
    const std::size_t length = std::strlen(prefix);
    if (std::strncmp(command, prefix, length) != 0) {
      return false;
    }
    const char *value = command + length;
    if (std::strcmp(value, "on") == 0 || std::strcmp(value, "off") == 0) {
      setter(std::strcmp(value, "on") == 0);
      ok_status();
    } else {
      error("expected on or off");
    }
    return true;
  };
  return toggle("touch ability ", &native_touch_set_ability_tap) ||
         toggle("touch calibrate ", &native_touch_set_calibrating);
}
