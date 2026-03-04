"""
V3 Plugin Architecture — Verification Test Script
Run: python tests/run_tests.py
"""
import sys, os, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

passed = 0
failed = 0

def check(name, condition):
    global passed, failed
    if condition:
        print(f"  [PASS] {name}")
        passed += 1
    else:
        print(f"  [FAIL] {name}")
        failed += 1

print("=" * 55)
print("  V3 Plugin Architecture — Verification Tests")
print("=" * 55)

# ─── 1. EventBus ───
print("\n--- EventBus ---")
from core.event_bus import EventBus, Events, make_event
EventBus.reset()
bus = EventBus.instance()

received = []
bus.subscribe('TestEvent', lambda e: received.append(e.payload['msg']))
bus.publish(make_event('TestEvent', msg='hello'))
check("pub/sub", received == ['hello'])

order = []
bus.subscribe('P', lambda e: order.append('low'), priority=0)
bus.subscribe('P', lambda e: order.append('high'), priority=10)
bus.publish(make_event('P'))
check("priority ordering", order == ['high', 'low'])

count = []
bus.subscribe('OS', lambda e: count.append(1), one_shot=True)
bus.publish(make_event('OS'))
bus.publish(make_event('OS'))
check("one-shot", len(count) == 1)

wild = []
bus.subscribe('*', lambda e: wild.append(e.name))
bus.publish(make_event('A'))
bus.publish(make_event('B'))
check("wildcard", 'A' in wild and 'B' in wild)

check("history", len(bus.get_history()) > 0)

b2 = EventBus.instance()
check("singleton", bus is b2)
bus.clear()

# ─── 2. Models ───
print("\n--- Models ---")
from core.models import Signal, FilterStep, Pipeline
sig = Signal(np.random.randn(4000), 8000.0)
check("Signal duration", abs(sig.duration - 0.5) < 0.01)
check("Signal channels", sig.n_channels == 1)

from scipy.signal import firwin
b = firwin(33, 1000, fs=8000)
step = FilterStep('LP', 'fir', coeffs_b=b, coeffs_a=np.array([1.0]))
pipe = Pipeline(steps=[step])
out = pipe.apply(sig)
check("Pipeline.apply length", len(out.samples) == 4000)
check("Pipeline.apply fs", out.fs == 8000)

# ─── 3. PluginManager ───
print("\n--- PluginManager ---")
from core.plugin_manager import PluginManager
EventBus.reset()
bus = EventBus.instance()
pm = PluginManager(bus)
manifests = pm.discover('plugins')
ids = [m.id for m in manifests]
check("discover 5 plugins", len(manifests) >= 5)
check("ai_assistant found", 'ai_assistant' in ids)
check("hil_twin found", 'hil_twin' in ids)
check("stress_testing found", 'stress_testing' in ids)
check("domain_toolkits found", 'domain_toolkits' in ids)
check("ux_ecosystem found", 'ux_ecosystem' in ids)

# ─── 4. AI Assistant ───
print("\n--- AI Assistant ---")
sys.path.insert(0, 'plugins/ai_assistant')
from plugins.ai_assistant.service import Service as AISvc
ai = AISvc()
ai.activate(bus, {})

steps = ai.parse_chat('notch 50Hz + lowpass 4000Hz')
check("chat: notch+lp", len(steps) >= 2)
check("chat: notch type", steps[0]['type'] == 'notch')
check("chat: notch freq", steps[0]['freq'] == 50)

freq = np.linspace(0, 4000, 4000)
mag = np.full_like(freq, -80.0)
for f0 in [50, 100, 150, 200]:
    mag[int(f0)] = -10
anomalies = ai.detect_anomalies(freq, mag, 8000)
types = [a.anomaly_type for a in anomalies]
check("anomaly detected", len(anomalies) > 0)

# ─── 5. HIL & Digital Twin ───
print("\n--- HIL & Digital Twin ---")
sys.path.insert(0, 'plugins/hil_twin')
from plugins.hil_twin.service import Service as HILSvc
hil = HILSvc()
hil.activate(bus, {})
resp = hil.push_and_test(b, np.array([1.0]), 8000, sig.samples)
check("MockDevice response", resp is not None and len(resp) == 4000)

res = hil.estimate_resources(b, np.array([1.0]), 8000)
check("CM4 MACs", res['cortex_m4'].macs_per_sample == len(b))
check("Zynq DSP48", res['zynq'].dsp_slices > 0)
check("Zynq LUT", res['zynq'].lut_count > 0)
print(f"  Info: CM4={res['cortex_m4'].macs_per_sample}MACs, "
      f"Zynq={res['zynq'].dsp_slices}DSP48/{res['zynq'].lut_count}LUT")

# ─── 6. Stress Testing ───
print("\n--- Stress Testing ---")
sys.path.insert(0, 'plugins/stress_testing')
from plugins.stress_testing.service import Service as StressSvc
st = StressSvc()
st.activate(bus, {})

report = st.monte_carlo_sensitivity(b, np.array([1.0]), 8000, n_trials=50)
check("MC method", report.method == 'monte_carlo')
check("MC pass_rate (FIR=1.0)", report.pass_rate == 1.0)
print(f"  Info: max_dev={report.max_deviation_db:.4f}dB, "
      f"p95={report.details['p95_dev_db']:.4f}dB")

fp = st.fixedpoint_sweep(b, np.array([1.0]), 8000, max_error_db=1.0)
check("FP found solution", len(fp.optimal_q) > 0 and fp.pass_rate > 0)
print(f"  Info: optimal Q-format = {fp.optimal_q}")

y_jit = st.simulate_jitter(sig.samples, 8000)
check("jitter sim", len(y_jit) == len(sig.samples))

y_clip = st.simulate_clipping(sig.samples, clip_level=0.5)
peak = np.max(np.abs(sig.samples))
check("clipping", np.max(np.abs(y_clip)) <= peak * 0.5 + 0.01)

y_loss = st.simulate_packet_loss(sig.samples, loss_rate=0.1)
check("packet loss", np.any(y_loss == 0))

# ─── 7. Domain Toolkits ───
print("\n--- Domain Toolkits ---")
sys.path.insert(0, 'plugins/domain_toolkits')
from plugins.domain_toolkits.service import Service as DomSvc
dom = DomSvc()
dom.activate(bus, {})

r = dom.third_octave(sig.samples, 8000)
check("1/3 octave bands", len(r['center_freqs']) > 0)

tone = np.sin(2 * np.pi * 1000 * np.arange(48000) / 48000)
thdn = dom.thd_plus_n(tone, 48000)
check("THD+N pure tone < 1%", thdn['thd_n_percent'] < 1)
print(f"  Info: THD+N = {thdn['thd_n_percent']:.4f}%")

eeg = np.sin(2*np.pi*10*np.arange(512)/256)
rhy = dom.eeg_rhythm_extraction(eeg, 256)
check("EEG alpha band", 'alpha' in rhy)

# ─── 8. UX & Ecosystem ───
print("\n--- UX & Ecosystem ---")
sys.path.insert(0, 'plugins/ux_ecosystem')
from plugins.ux_ecosystem.service import Service as UXSvc
ux = UXSvc()
ux.activate(bus, {})

s3d = ux.compute_3d_spectrogram(sig.samples, 8000, nperseg=128)
check("3D spectrogram 2D", s3d['magnitude_db'].ndim == 2)
print(f"  Info: shape = {s3d['magnitude_db'].shape}")

reg = ux.get_registry()
check("registry entries", len(reg) >= 5)

jid = ux.submit_job('test', {})
time.sleep(1.5)
status = ux.get_job_status(jid)
check("cloud job done", status['status'] == 'done')

# ─── Summary ───
print("\n" + "=" * 55)
total = passed + failed
print(f"  Results: {passed}/{total} passed, {failed} failed")
if failed == 0:
    print("  ALL TESTS PASSED")
else:
    print(f"  {failed} TESTS FAILED")
print("=" * 55)
sys.exit(0 if failed == 0 else 1)
