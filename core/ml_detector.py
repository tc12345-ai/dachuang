"""
ML Transient Event Detector — 机器学习瞬态事件检测模块

Statistical and ML-based methods for detecting transient events
in signal data: threshold, energy burst, spectral anomaly, and
optional sklearn-based classifier.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable
from scipy import signal as sig


@dataclass
class TransientEvent:
    """Detected transient event / 检测到的瞬态事件."""
    start_sample: int = 0
    end_sample: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0
    peak_amplitude: float = 0.0
    energy: float = 0.0
    event_type: str = 'unknown'
    confidence: float = 0.0
    features: dict = field(default_factory=dict)


@dataclass
class DetectionResult:
    """Detection result / 检测结果."""
    events: List[TransientEvent] = field(default_factory=list)
    n_events: int = 0
    detection_signal: np.ndarray = None   # Detection metric over time
    threshold_line: np.ndarray = None     # Threshold over time
    method: str = ''
    info: str = ''


class TransientDetector:
    """
    Transient Event Detector — 瞬态事件检测器
    
    Multiple detection methods:
    1. Threshold-based (amplitude/envelope)
    2. Energy burst detection (short-time energy)
    3. Spectral anomaly (spectral flux / novelty)
    4. Statistical (z-score, CUSUM)
    5. ML-based (sklearn classifier, optional)
    """
    
    METHODS = ['threshold', 'energy', 'spectral_flux',
               'zscore', 'cusum', 'ml_classifier']
    
    def __init__(self):
        self._ml_model = None
        self._ml_scaler = None
    
    def detect(self, data: np.ndarray, fs: float,
               method: str = 'energy',
               threshold: float = None,
               min_duration_ms: float = 1.0,
               merge_gap_ms: float = 5.0,
               **kwargs) -> DetectionResult:
        """
        Detect transient events.
        
        Args:
            data: Input signal (1D)
            fs: Sampling rate
            method: Detection method name
            threshold: Detection threshold (None = auto)
            min_duration_ms: Min event duration in ms
            merge_gap_ms: Merge events closer than this
            **kwargs: Method-specific parameters
        Returns:
            DetectionResult
        """
        x = np.asarray(data, dtype=np.float64).ravel()
        result = DetectionResult(method=method)
        
        if method == 'threshold':
            det_signal = self._threshold_detect(x, fs, **kwargs)
        elif method == 'energy':
            det_signal = self._energy_detect(x, fs, **kwargs)
        elif method == 'spectral_flux':
            det_signal = self._spectral_flux_detect(x, fs, **kwargs)
        elif method == 'zscore':
            det_signal = self._zscore_detect(x, fs, **kwargs)
        elif method == 'cusum':
            det_signal = self._cusum_detect(x, fs, **kwargs)
        elif method == 'ml_classifier':
            return self._ml_detect(x, fs, **kwargs)
        else:
            det_signal = self._energy_detect(x, fs)
        
        result.detection_signal = det_signal
        
        # Auto threshold
        if threshold is None:
            noise_est = np.median(det_signal)
            threshold = noise_est + 3.0 * np.std(det_signal)
        
        result.threshold_line = np.full_like(det_signal, threshold)
        
        # Find events (regions above threshold)
        above = det_signal > threshold
        min_samples = max(1, int(min_duration_ms * fs / 1000))
        merge_samples = max(1, int(merge_gap_ms * fs / 1000))
        
        events = self._extract_events(x, above, fs, min_samples, merge_samples)
        result.events = events
        result.n_events = len(events)
        result.info = f"{method}: {len(events)} events detected, threshold={threshold:.4f}"
        
        return result
    
    def _threshold_detect(self, x, fs, **kwargs):
        """Simple amplitude envelope detection."""
        # Compute envelope using Hilbert transform
        analytic = sig.hilbert(x)
        envelope = np.abs(analytic)
        
        # Smooth envelope
        win_ms = kwargs.get('window_ms', 5.0)
        win_samples = max(3, int(win_ms * fs / 1000))
        if win_samples % 2 == 0:
            win_samples += 1
        envelope = np.convolve(envelope, np.ones(win_samples)/win_samples, 'same')
        
        return envelope
    
    def _energy_detect(self, x, fs, **kwargs):
        """Short-time energy detection."""
        frame_ms = kwargs.get('frame_ms', 10.0)
        hop_ms = kwargs.get('hop_ms', 2.5)
        
        frame_len = max(4, int(frame_ms * fs / 1000))
        hop_len = max(1, int(hop_ms * fs / 1000))
        
        n = len(x)
        n_frames = max(1, (n - frame_len) // hop_len + 1)
        
        energy = np.zeros(n)
        for i in range(n_frames):
            start = i * hop_len
            end = min(start + frame_len, n)
            frame_energy = np.sum(x[start:end] ** 2) / frame_len
            energy[start:end] = np.maximum(energy[start:end], frame_energy)
        
        # Convert to dB
        energy_db = 10 * np.log10(np.maximum(energy, 1e-20))
        
        return energy_db
    
    def _spectral_flux_detect(self, x, fs, **kwargs):
        """Spectral flux novelty detection."""
        nperseg = kwargs.get('nperseg', 256)
        noverlap = kwargs.get('noverlap', nperseg * 3 // 4)
        
        f, t, Zxx = sig.stft(x, fs=fs, nperseg=nperseg, noverlap=noverlap)
        mag = np.abs(Zxx)
        
        # Spectral flux: sum of positive differences
        flux = np.zeros(mag.shape[1])
        for i in range(1, mag.shape[1]):
            diff = mag[:, i] - mag[:, i-1]
            flux[i] = np.sum(np.maximum(diff, 0) ** 2)
        
        # Interpolate back to signal length
        flux_interp = np.interp(
            np.arange(len(x)) / fs, t, flux)
        
        return flux_interp
    
    def _zscore_detect(self, x, fs, **kwargs):
        """Z-score based anomaly detection."""
        window_ms = kwargs.get('window_ms', 100.0)
        win = max(10, int(window_ms * fs / 1000))
        
        n = len(x)
        zscore = np.zeros(n)
        
        for i in range(win, n):
            segment = x[max(0, i-win):i]
            mu = np.mean(segment)
            sigma = np.std(segment)
            if sigma > 1e-12:
                zscore[i] = abs(x[i] - mu) / sigma
        
        return zscore
    
    def _cusum_detect(self, x, fs, **kwargs):
        """CUSUM (Cumulative Sum) change detection."""
        drift = kwargs.get('drift', 0.5)
        
        x_centered = x - np.mean(x)
        n = len(x)
        
        g_pos = np.zeros(n)
        g_neg = np.zeros(n)
        
        for i in range(1, n):
            g_pos[i] = max(0, g_pos[i-1] + x_centered[i] - drift)
            g_neg[i] = max(0, g_neg[i-1] - x_centered[i] - drift)
        
        return g_pos + g_neg
    
    def _ml_detect(self, x, fs, **kwargs) -> DetectionResult:
        """ML-based detection using sklearn."""
        result = DetectionResult(method='ml_classifier')
        
        # Extract features per frame
        frame_ms = kwargs.get('frame_ms', 20.0)
        hop_ms = kwargs.get('hop_ms', 10.0)
        frame_len = max(4, int(frame_ms * fs / 1000))
        hop_len = max(1, int(hop_ms * fs / 1000))
        
        n = len(x)
        features_list = []
        frame_starts = []
        
        for start in range(0, n - frame_len, hop_len):
            frame = x[start:start + frame_len]
            feats = self._extract_frame_features(frame, fs)
            features_list.append(feats)
            frame_starts.append(start)
        
        if not features_list:
            result.info = "Signal too short for ML detection"
            return result
        
        feature_matrix = np.array(features_list)
        
        if self._ml_model is not None:
            # Use trained model
            if self._ml_scaler:
                feature_matrix = self._ml_scaler.transform(feature_matrix)
            predictions = self._ml_model.predict(feature_matrix)
            proba = self._ml_model.predict_proba(feature_matrix)[:, 1] \
                if hasattr(self._ml_model, 'predict_proba') else predictions.astype(float)
        else:
            # Unsupervised: use Isolation Forest
            try:
                from sklearn.ensemble import IsolationForest
                from sklearn.preprocessing import StandardScaler
                
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(feature_matrix)
                
                clf = IsolationForest(contamination=0.05, random_state=42)
                predictions = clf.fit_predict(X_scaled)
                predictions = (predictions == -1).astype(int)  # -1 = anomaly
                scores = -clf.decision_function(X_scaled)
                proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
                
            except ImportError:
                # Fallback to energy-based
                result.info = "sklearn not installed, using energy detection"
                return self.detect(x, fs, method='energy')
        
        # Build detection signal (interpolate probabilities)
        det_signal = np.zeros(n)
        for i, start in enumerate(frame_starts):
            end = min(start + frame_len, n)
            det_signal[start:end] = np.maximum(det_signal[start:end], proba[i])
        
        result.detection_signal = det_signal
        result.threshold_line = np.full(n, 0.5)
        
        # Extract events
        above = det_signal > 0.5
        min_samples = max(1, int(1.0 * fs / 1000))
        events = self._extract_events(x, above, fs, min_samples, int(5 * fs / 1000))
        
        for evt in events:
            evt.event_type = 'ml_anomaly'
        
        result.events = events
        result.n_events = len(events)
        result.info = f"ML detection: {len(events)} events, features={feature_matrix.shape[1]}"
        
        return result
    
    def _extract_frame_features(self, frame, fs):
        """Extract features from a signal frame for ML."""
        feats = []
        
        # Time-domain features
        feats.append(np.mean(np.abs(frame)))       # Mean absolute
        feats.append(np.std(frame))                 # Std dev
        feats.append(np.max(np.abs(frame)))         # Peak
        feats.append(np.sqrt(np.mean(frame ** 2)))  # RMS
        
        # Zero crossing rate
        zcr = np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * len(frame))
        feats.append(zcr)
        
        # Spectral features
        X = np.abs(np.fft.rfft(frame))
        freqs = np.fft.rfftfreq(len(frame), 1/fs)
        
        total_power = np.sum(X ** 2)
        if total_power > 0:
            spectral_centroid = np.sum(freqs * X ** 2) / total_power
            spectral_spread = np.sqrt(
                np.sum((freqs - spectral_centroid) ** 2 * X ** 2) / total_power)
        else:
            spectral_centroid = 0
            spectral_spread = 0
        
        feats.append(spectral_centroid)
        feats.append(spectral_spread)
        
        # Spectral rolloff (85%)
        cumsum = np.cumsum(X ** 2)
        rolloff_idx = np.searchsorted(cumsum, 0.85 * total_power)
        feats.append(freqs[min(rolloff_idx, len(freqs)-1)])
        
        # Kurtosis
        n = len(frame)
        mu = np.mean(frame)
        sigma = np.std(frame)
        if sigma > 0:
            kurtosis = np.mean(((frame - mu) / sigma) ** 4) - 3
        else:
            kurtosis = 0
        feats.append(kurtosis)
        
        # Crest factor
        rms = np.sqrt(np.mean(frame ** 2))
        crest = np.max(np.abs(frame)) / (rms + 1e-12)
        feats.append(crest)
        
        return feats
    
    def train_classifier(self, normal_data: np.ndarray,
                         anomaly_data: np.ndarray,
                         fs: float, frame_ms: float = 20.0,
                         hop_ms: float = 10.0):
        """
        Train ML classifier for transient detection.
        
        Args:
            normal_data: Normal signal (no transients)
            anomaly_data: Signal with transients
            fs: Sampling rate
        """
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import StandardScaler
            
            frame_len = int(frame_ms * fs / 1000)
            hop_len = int(hop_ms * fs / 1000)
            
            # Extract features
            X_normal = []
            for start in range(0, len(normal_data) - frame_len, hop_len):
                frame = normal_data[start:start + frame_len]
                X_normal.append(self._extract_frame_features(frame, fs))
            
            X_anomaly = []
            for start in range(0, len(anomaly_data) - frame_len, hop_len):
                frame = anomaly_data[start:start + frame_len]
                X_anomaly.append(self._extract_frame_features(frame, fs))
            
            X = np.array(X_normal + X_anomaly)
            y = np.array([0] * len(X_normal) + [1] * len(X_anomaly))
            
            self._ml_scaler = StandardScaler()
            X_scaled = self._ml_scaler.fit_transform(X)
            
            self._ml_model = RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42)
            self._ml_model.fit(X_scaled, y)
            
            return True
        except ImportError:
            print("sklearn not installed for ML training")
            return False
    
    def _extract_events(self, x, above, fs, min_samples, merge_samples):
        """Extract event segments from boolean mask."""
        events = []
        n = len(above)
        
        # Find contiguous regions
        diff = np.diff(above.astype(int))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1
        
        if above[0]:
            starts = np.concatenate([[0], starts])
        if above[-1]:
            ends = np.concatenate([ends, [n]])
        
        if len(starts) == 0 or len(ends) == 0:
            return events
        
        # Pair starts and ends
        n_events = min(len(starts), len(ends))
        
        # Merge close events
        merged_starts = [starts[0]]
        merged_ends = [ends[0]]
        
        for i in range(1, n_events):
            if starts[i] - merged_ends[-1] < merge_samples:
                merged_ends[-1] = ends[i]
            else:
                merged_starts.append(starts[i])
                merged_ends.append(ends[i])
        
        for s, e in zip(merged_starts, merged_ends):
            if e - s < min_samples:
                continue
            
            segment = x[s:e]
            evt = TransientEvent(
                start_sample=int(s),
                end_sample=int(e),
                start_time=s / fs,
                end_time=e / fs,
                duration=(e - s) / fs,
                peak_amplitude=float(np.max(np.abs(segment))),
                energy=float(np.sum(segment ** 2)),
                event_type='transient',
                confidence=0.8,
            )
            events.append(evt)
        
        return events
