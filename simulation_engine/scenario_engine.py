from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class Event:
  """
  Represents a scenario event that can be injected into the simulation.

  name: Unique event type/name, e.g., "flood", "fx_shock", "cofin_delay".
  time_step: Simulation time step when the event becomes effective.
  payload: Arbitrary structured data describing parameters and intended effects.
  targets: Optional list of agent identifiers that should receive the event.
  """
  name: str
  time_step: int
  payload: Dict[str, Any] = field(default_factory=dict)
  targets: Optional[List[str]] = None


class EventBus:
  """
  Minimal event bus for scheduling and dispatching scenario events.

  Features:
    - subscribe(event_name, handler): register handler for specific event name
      or wildcard "*" to receive all events.
    - schedule(event): queue event at event.time_step
    - inject_now(event): dispatch immediately at current time
    - tick(steps): advance time and dispatch due events
    - advance_to(time_step): jump forward and dispatch all due events
  """

  def __init__(self) -> None:
    self._now: int = 0
    self._queue: Dict[int, List[Event]] = {}
    self._subscribers: Dict[str, List[Callable[[Event], None]]] = {}

  @property
  def now(self) -> int:
    return self._now

  def subscribe(self, event_name: str, handler: Callable[[Event], None]) -> None:
    if event_name not in self._subscribers:
      self._subscribers[event_name] = []
    self._subscribers[event_name].append(handler)

  def unsubscribe(self, event_name: str, handler: Callable[[Event], None]) -> None:
    if event_name in self._subscribers:
      self._subscribers[event_name] = [h for h in self._subscribers[event_name] if h != handler]
      if not self._subscribers[event_name]:
        del self._subscribers[event_name]

  def schedule(self, event: Event) -> None:
    if event.time_step not in self._queue:
      self._queue[event.time_step] = []
    self._queue[event.time_step].append(event)

  def inject_now(self, event: Event) -> None:
    # Normalize event time to current time, but dispatch immediately
    event.time_step = self._now
    self._dispatch(event)

  def tick(self, steps: int = 1) -> None:
    for _ in range(max(steps, 0)):
      self._now += 1
      self._dispatch_due()

  def advance_to(self, time_step: int) -> None:
    if time_step < self._now:
      return
    while self._now < time_step:
      self._now += 1
      self._dispatch_due()

  # Internal helpers
  def _dispatch_due(self) -> None:
    if not self._queue:
      return
    due_times = [t for t in self._queue.keys() if t <= self._now]
    if not due_times:
      return
    for t in sorted(due_times):
      events = self._queue.pop(t, [])
      for ev in events:
        self._dispatch(ev)

  def _dispatch(self, event: Event) -> None:
    # Specific subscribers
    for handler in self._subscribers.get(event.name, []):
      handler(event)
    # Wildcard subscribers
    for handler in self._subscribers.get("*", []):
      handler(event)


class ScenarioEngine:
  """
  Coordinates scenario events and routes them to registered agents.

  Notes:
    - Agents are expected to expose a `.remember(text: str, time_step: int = 0)`
      method (compatible with GenerativeAgent) so events can be persisted in
      their memory stream as grounded context for downstream behavior.
  """

  def __init__(self, agents: Optional[Dict[str, Any]] = None) -> None:
    self.bus = EventBus()
    self.agents: Dict[str, Any] = agents or {}

    # Default subscriber: write events to targeted agents' memory
    self.bus.subscribe("*", self._record_event_to_agents)

  def register_agent(self, agent_id: str, agent: Any) -> None:
    self.agents[agent_id] = agent

  def unregister_agent(self, agent_id: str) -> None:
    if agent_id in self.agents:
      del self.agents[agent_id]

  def inject_now(self, name: str, payload: Optional[Dict[str, Any]] = None,
                 targets: Optional[List[str]] = None) -> Event:
    ev = Event(name=name, time_step=self.bus.now, payload=payload or {}, targets=targets)
    self.bus.inject_now(ev)
    return ev

  def schedule(self, name: str, in_steps: int, payload: Optional[Dict[str, Any]] = None,
               targets: Optional[List[str]] = None) -> Event:
    ev = Event(name=name, time_step=self.bus.now + max(in_steps, 0), payload=payload or {}, targets=targets)
    self.bus.schedule(ev)
    return ev

  def tick(self, steps: int = 1) -> None:
    self.bus.tick(steps)

  def advance_to(self, time_step: int) -> None:
    self.bus.advance_to(time_step)

  # Subscriber: record every event into targets' memory
  def _record_event_to_agents(self, event: Event) -> None:
    target_ids = event.targets if event.targets else list(self.agents.keys())
    if not target_ids:
      return
    memory_text = self._format_memory_text(event)
    for agent_id in target_ids:
      agent = self.agents.get(agent_id)
      if not agent:
        continue
      try:
        # Agents may accept an optional time_step argument; fall back if not
        agent.remember(memory_text, time_step=self.bus.now)  # type: ignore[arg-type]
      except TypeError:
        agent.remember(memory_text)  # type: ignore[call-arg]

  def _format_memory_text(self, event: Event) -> str:
    # Keep memory strings concise but structured for retrieval
    payload_preview = {k: event.payload[k] for k in list(event.payload)[:10]}
    return (
      f"Event[{event.name}] @t={self.bus.now} "
      f"targets={event.targets or 'ALL'} "
      f"payload={payload_preview}"
    )


__all__ = [
  "Event",
  "EventBus",
  "ScenarioEngine",
]

