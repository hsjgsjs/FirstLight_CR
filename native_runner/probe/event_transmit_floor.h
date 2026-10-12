#pragma once

#include <cstdint>

// Live mode sends only the events after the caller's cursors (everything retained otherwise).
// Set by the control thread and read by the game thread, both under g_control_mutex; zero
// outside a live capture, so every other observation path transmits the whole retained ring.
enum EventTransmitRing { kTransmitCombat, kTransmitPhase, kTransmitVisibility, kTransmitRemaining, kTransmitRings };
inline std::uint64_t g_event_transmit_floor[kTransmitRings] = {};

inline std::uint64_t event_transmit_from(EventTransmitRing ring, std::uint64_t retained_oldest, std::uint64_t next) {
  const std::uint64_t floor = g_event_transmit_floor[ring];
  if (floor <= retained_oldest) {
    return retained_oldest;
  }
  return floor < next ? floor : next;
}
