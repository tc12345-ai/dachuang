"""
Tests for EventBus — 事件总线测试
"""

import pytest
from core.event_bus import EventBus, Events, Event, make_event


class TestEventBus:
    def test_publish_subscribe(self, bus):
        received = []
        bus.subscribe('TestEvent', lambda e: received.append(e))
        bus.publish(make_event('TestEvent', msg='hello'))
        assert len(received) == 1
        assert received[0].payload['msg'] == 'hello'

    def test_multiple_subscribers(self, bus):
        received = []
        bus.subscribe('E', lambda e: received.append('A'))
        bus.subscribe('E', lambda e: received.append('B'))
        bus.publish(make_event('E'))
        assert received == ['A', 'B']

    def test_priority(self, bus):
        received = []
        bus.subscribe('E', lambda e: received.append('low'), priority=0)
        bus.subscribe('E', lambda e: received.append('high'), priority=10)
        bus.publish(make_event('E'))
        assert received == ['high', 'low']

    def test_unsubscribe_by_id(self, bus):
        received = []
        bus.subscribe('E', lambda e: received.append(1),
                      subscriber_id='plugin_a')
        bus.unsubscribe('E', subscriber_id='plugin_a')
        bus.publish(make_event('E'))
        assert len(received) == 0

    def test_unsubscribe_all(self, bus):
        received = []
        bus.subscribe('A', lambda e: received.append(1), subscriber_id='p')
        bus.subscribe('B', lambda e: received.append(2), subscriber_id='p')
        bus.unsubscribe_all('p')
        bus.publish(make_event('A'))
        bus.publish(make_event('B'))
        assert len(received) == 0

    def test_one_shot(self, bus):
        received = []
        bus.subscribe('E', lambda e: received.append(1), one_shot=True)
        bus.publish(make_event('E'))
        bus.publish(make_event('E'))
        assert len(received) == 1

    def test_wildcard(self, bus):
        received = []
        bus.subscribe('*', lambda e: received.append(e.name))
        bus.publish(make_event('A'))
        bus.publish(make_event('B'))
        assert received == ['A', 'B']

    def test_mute_unmute(self, bus):
        received = []
        bus.subscribe('E', lambda e: received.append(1))
        bus.mute('E')
        bus.publish(make_event('E'))
        assert len(received) == 0
        bus.unmute('E')
        bus.publish(make_event('E'))
        assert len(received) == 1

    def test_history(self, bus):
        bus.publish(make_event('A'))
        bus.publish(make_event('B'))
        bus.publish(make_event('A'))
        hist = bus.get_history('A')
        assert len(hist) == 2
        all_hist = bus.get_history()
        assert len(all_hist) == 3

    def test_subscriber_count(self, bus):
        bus.subscribe('A', lambda e: None)
        bus.subscribe('A', lambda e: None)
        bus.subscribe('B', lambda e: None)
        assert bus.subscriber_count('A') == 2
        assert bus.subscriber_count() == 3

    def test_singleton(self):
        from core.event_bus import EventBus
        b1 = EventBus.instance()
        b2 = EventBus.instance()
        assert b1 is b2

    def test_exception_doesnt_break(self, bus):
        """Subscriber exception should not prevent other subscribers."""
        received = []
        bus.subscribe('E', lambda e: 1/0, priority=10)  # Will raise
        bus.subscribe('E', lambda e: received.append(1), priority=0)
        bus.publish(make_event('E'))
        assert len(received) == 1
