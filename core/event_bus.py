"""
EventBus — 统一事件总线

Thread-safe publish/subscribe event system for inter-module communication.
All plugins communicate through events, never by direct coupling.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set
from collections import defaultdict
import traceback
import numpy as np


# ═══════════════════════════════════════════════════
#  Event Definitions
# ═══════════════════════════════════════════════════

@dataclass
class Event:
    """Base event class — 所有事件的基类."""
    name: str
    payload: Dict[str, Any] = field(default_factory=dict)
    source: str = ''            # Plugin/module that fired the event
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


# Concrete event factory for typed events
def make_event(name: str, source: str = '', **payload) -> Event:
    return Event(name=name, payload=payload, source=source)


# ─── Standard Event Names ───
class Events:
    """
    Standard event names — 标准事件名

    Payload schemas (dict keys):
      SignalLoaded:     signal: Signal, filepath: str
      SegmentLabeled:   signal: Signal, start: int, end: int, label: str
      SpectrumComputed: freq_hz: ndarray, magnitude_db: ndarray, method: str
      FilterDesigned:   b: ndarray, a: ndarray, info: str, spec: FilterSpec
      PipelineApplied:  input: Signal, output: Signal, pipeline: Pipeline
      CodeGenerated:    language: str, code: str
      MeasurementDone:  metrics: dict
      PluginLoaded:     plugin_id: str, name: str
      PluginUnloaded:   plugin_id: str
      ApiStarted:       url: str
      ApiStopped:       (empty)
      StreamFrame:      frame: ndarray, fs: float
      AnomalyDetected:  anomalies: list[dict], signal_id: str
      FilterRecommended: recommendations: list[dict]
      HilCoeffPushed:   device: str, coeffs: ndarray
      HilResponseRecv:  device: str, response: ndarray
      StressTestDone:   report: dict
      ChatCommand:      text: str
      ChatResponse:     text: str, actions: list
      Spectrum3DReady:  data: ndarray, freq: ndarray, time: ndarray
    """
    SIGNAL_LOADED       = 'SignalLoaded'
    SEGMENT_LABELED     = 'SegmentLabeled'
    SPECTRUM_COMPUTED   = 'SpectrumComputed'
    FILTER_DESIGNED     = 'FilterDesigned'
    PIPELINE_APPLIED    = 'PipelineApplied'
    CODE_GENERATED      = 'CodeGenerated'
    MEASUREMENT_DONE    = 'MeasurementDone'
    PLUGIN_LOADED       = 'PluginLoaded'
    PLUGIN_UNLOADED     = 'PluginUnloaded'
    API_STARTED         = 'ApiStarted'
    API_STOPPED         = 'ApiStopped'
    STREAM_FRAME        = 'StreamFrame'
    ANOMALY_DETECTED    = 'AnomalyDetected'
    FILTER_RECOMMENDED  = 'FilterRecommended'
    HIL_COEFF_PUSHED    = 'HilCoeffPushed'
    HIL_RESPONSE_RECV   = 'HilResponseRecv'
    STRESS_TEST_DONE    = 'StressTestDone'
    CHAT_COMMAND        = 'ChatCommand'
    CHAT_RESPONSE       = 'ChatResponse'
    SPECTRUM_3D_READY   = 'Spectrum3DReady'
    RESOURCE_ESTIMATED  = 'ResourceEstimated'
    DOMAIN_RESULT       = 'DomainResult'


# ═══════════════════════════════════════════════════
#  Subscription Record
# ═══════════════════════════════════════════════════

@dataclass
class _Subscription:
    callback: Callable[[Event], None]
    subscriber_id: str
    priority: int = 0           # Higher = called first
    one_shot: bool = False      # Auto-unsubscribe after first call


# ═══════════════════════════════════════════════════
#  EventBus (singleton-ready)
# ═══════════════════════════════════════════════════

class EventBus:
    """
    Thread-safe EventBus — 线程安全事件总线

    Usage:
        bus = EventBus()
        bus.subscribe('SignalLoaded', my_handler)
        bus.publish(make_event('SignalLoaded', signal=sig))
    """
    _instance: Optional['EventBus'] = None
    _lock_singleton = threading.Lock()

    @classmethod
    def instance(cls) -> 'EventBus':
        """Get or create _the_ global singleton."""
        if cls._instance is None:
            with cls._lock_singleton:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls):
        """Reset singleton (for testing)."""
        cls._instance = None

    def __init__(self):
        self._subscribers: Dict[str, List[_Subscription]] = defaultdict(list)
        self._lock = threading.RLock()
        self._history: List[Event] = []
        self._max_history = 200
        self._muted: Set[str] = set()          # Muted event names

    # ─── subscribe / unsubscribe ───

    def subscribe(self, event_name: str, callback: Callable[[Event], None],
                  subscriber_id: str = '', priority: int = 0,
                  one_shot: bool = False):
        """
        Subscribe to an event.

        Args:
            event_name: Event name to listen for (or '*' for all)
            callback: func(Event) -> None
            subscriber_id: Identifier (for unsubscribe-by-id)
            priority: Higher = called earlier
            one_shot: Auto-remove after first invocation
        """
        sub = _Subscription(callback=callback, subscriber_id=subscriber_id,
                            priority=priority, one_shot=one_shot)
        with self._lock:
            self._subscribers[event_name].append(sub)
            self._subscribers[event_name].sort(key=lambda s: -s.priority)

    def unsubscribe(self, event_name: str, callback: Callable = None,
                    subscriber_id: str = ''):
        """Remove subscriptions by callback reference or subscriber_id."""
        with self._lock:
            subs = self._subscribers.get(event_name, [])
            self._subscribers[event_name] = [
                s for s in subs
                if not ((callback and s.callback is callback) or
                        (subscriber_id and s.subscriber_id == subscriber_id))
            ]

    def unsubscribe_all(self, subscriber_id: str):
        """Remove ALL subscriptions for a subscriber (plugin teardown)."""
        with self._lock:
            for name in list(self._subscribers.keys()):
                self._subscribers[name] = [
                    s for s in self._subscribers[name]
                    if s.subscriber_id != subscriber_id
                ]

    # ─── publish ───

    def publish(self, event: Event):
        """
        Publish an event synchronously.
        All subscribers are called in priority order on the calling thread.
        """
        if event.name in self._muted:
            return

        with self._lock:
            # Wildcard subscribers + named subscribers
            targets = (list(self._subscribers.get('*', []))
                       + list(self._subscribers.get(event.name, [])))
            targets.sort(key=lambda s: -s.priority)

        one_shots = []
        for sub in targets:
            try:
                sub.callback(event)
            except Exception:
                traceback.print_exc()
            if sub.one_shot:
                one_shots.append(sub)

        # Cleanup one-shots
        if one_shots:
            with self._lock:
                for sub in one_shots:
                    subs = self._subscribers.get(event.name, [])
                    if sub in subs:
                        subs.remove(sub)

        # History
        with self._lock:
            self._history.append(event)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]

    def publish_async(self, event: Event):
        """Publish event on a background thread."""
        t = threading.Thread(target=self.publish, args=(event,), daemon=True)
        t.start()

    # ─── utilities ───

    def mute(self, event_name: str):
        self._muted.add(event_name)

    def unmute(self, event_name: str):
        self._muted.discard(event_name)

    def get_history(self, event_name: str = None,
                    last_n: int = 50) -> List[Event]:
        with self._lock:
            if event_name:
                return [e for e in self._history
                        if e.name == event_name][-last_n:]
            return list(self._history[-last_n:])

    def subscriber_count(self, event_name: str = None) -> int:
        with self._lock:
            if event_name:
                return len(self._subscribers.get(event_name, []))
            return sum(len(v) for v in self._subscribers.values())

    def clear(self):
        """Clear all subscriptions and history."""
        with self._lock:
            self._subscribers.clear()
            self._history.clear()
            self._muted.clear()
