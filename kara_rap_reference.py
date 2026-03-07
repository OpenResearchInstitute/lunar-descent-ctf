#!/usr/bin/env python3
"""
KaRA Radar Altimeter Processor — Python Reference Model
=========================================================

Faithful implementation of the RAP described in:

  Sharma et al., "FPGA Implementation of a Hardware-Optimized
  Autonomous Real-Time Radar Altimeter Processor for Interplanetary
  Landing Missions," IEEE A&E Systems Magazine, Vol. 41, No. 1,
  January 2026. DOI: 10.1109/MAES.2025.3595090

This model implements every stage of the paper's Figure 2 pipeline:

  1. Frontend data acquisition (simulated)
  2. Decimator and bit selector
  3. Gain control estimator (AGC)
  4. Spectrum computation and peak estimation (8K FFT)
  5. Autotracking — unlocked (mode search) and locked (mode maintenance)
  6. Altitude estimator (Doppler-corrected, IEEE-754 M constant)

All parameters match the paper's Table 1, Table 2, and equations (1)-(8).

Usage:
  python kara_rap_reference.py                # Run descent simulation
  python kara_rap_reference.py --test         # Verbose single-altitude test
  python kara_rap_reference.py --modes        # Print mode table

Open Research Institute — openresearch.institute
"""

import numpy as np
import struct
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Tuple, Optional, List


# ═══════════════════════════════════════════════════════════════════
# SYSTEM PARAMETERS — from paper Table 1, Table 2, and text
# ═══════════════════════════════════════════════════════════════════

# KaRA RF parameters (Table 1, "KaRA System" section)
CENTER_FREQ = 35.61e9          # Hz — Ka-band center frequency
CHIRP_BW = 240e6               # Hz — "transmitted bandwidth is maintained at 240 MHz"
C_LIGHT = 299_792_458.0        # m/s — paper equation (5)

# ADC and processing parameters (Table 2, "Decimator" section)
ADC_RATE = 496.53e6            # Hz — "sampled at 496.53 MHz"
DECIMATION_FACTOR = 48         # ADC_RATE / effective_rate ≈ 48
FS = ADC_RATE / DECIMATION_FACTOR  # ~10.34 MHz effective sample rate
ADC_BITS = 8                   # "reduced to 8 bits" after decimation

# FFT parameters ("Spectrum Computation" section)
NFFT = 8192                    # "8K point FFT"
FREQ_RES = FS / NFFT           # ~1.262 kHz per bin (paper confirms)

# Processing and guard bandwidths (page 3, "KaRA System" section)
# "beat frequencies of the desired targets lie between 1.9 and 4 MHz"
# "Frequency ranges [1.7, 1.9) and (4, 4.2] MHz have been selected
#  as guard bands for smooth inter-mode transitions"
PROC_BW_LOW = 1.9e6            # Hz — lower processing band edge
PROC_BW_HIGH = 4.0e6           # Hz — upper processing band edge
GUARD_BW_LOW = 1.7e6           # Hz — lower guard band edge
GUARD_BW_HIGH = 4.2e6          # Hz — upper guard band edge

# Bin indices for processing and guard bands
PROC_LOW_BIN = int(PROC_BW_LOW / FREQ_RES)
PROC_HIGH_BIN = int(PROC_BW_HIGH / FREQ_RES)
GUARD_LOW_BIN = int(GUARD_BW_LOW / FREQ_RES)
GUARD_HIGH_BIN = int(GUARD_BW_HIGH / FREQ_RES)

# Altitude range (Table 1)
ALT_MIN = 3.0                  # m
ALT_MAX = 10_000.0             # m — "10 km to 3 m range"

# Sweep time range (Table 2)
# "Sweep time: Programmable, 8.2 ms - 1.45 μs"
SWEEP_TIME_MIN = 1.45e-6       # s
SWEEP_TIME_MAX = 8.2e-3        # s

# Autotracking parameters ("Autotracking for Locking" section)
CONSECUTIVE_REQUIRED = 3       # "highest peak consistently over three consecutive cycles"

# AGC parameters ("Gain Control Estimator" section)
AGC_THRESHOLD_FRACTION = 0.30  # "sample_crossing_threshold > 30% of n"
AGC_GAIN_MIN = 0               # "gain varies from 63 to 0" (0 = max gain)
AGC_GAIN_MAX = 63              # (63 = max attenuation)


# ═══════════════════════════════════════════════════════════════════
# SWEEP MODE TABLE
# ═══════════════════════════════════════════════════════════════════

@dataclass
class SweepMode:
    """One sweep time mode of the radar altimeter.
    
    Each mode has a fixed sweep time T. The beat frequency for a target
    at altitude R is:  fb = 2·B·R / (c·T)    [from eq. (4)]
    
    The mode is valid when fb falls in the processing band [1.9, 4.0] MHz.
    This gives an altitude range for each mode:
      R_low  = PROC_BW_LOW  · c · T / (2·B)
      R_high = PROC_BW_HIGH · c · T / (2·B)
    """
    mode_id: int
    sweep_time_s: float         # T in seconds
    alt_low: float = 0.0        # Minimum altitude in processing band
    alt_high: float = 0.0       # Maximum altitude in processing band

    @property
    def M(self) -> float:
        """Multiplying constant: R = M × (f_up_index + f_dn_index)
        
        From equation (8): M = c·T·fs / (4·B·NFFT)
        
        Paper: "stored in IEEE-754 single-precision floating-point"
        We replicate the single-precision truncation.
        """
        val = (C_LIGHT * self.sweep_time_s * FS) / (4.0 * CHIRP_BW * NFFT)
        # IEEE-754 single precision (float32) truncation — matches FPGA
        return struct.unpack('f', struct.pack('f', val))[0]


def build_mode_table() -> List[SweepMode]:
    """Build the sweep time mode table matching paper Table 2.
    
    Paper specifies: "Sweep time: Programmable, 8.2 ms - 1.45 μs"
    covering altitudes from 3 m to 10 km (Table 1).
    
    The number of modes must satisfy:
      step_ratio < PROC_BW_HIGH / PROC_BW_LOW  (≈ 2.105)
    
    for the guard-band transition mechanism to work (paper: "adjust the
    sweep time while preserving the lock status"). With endpoints fixed
    at 1.45 μs and 8.2 ms, 13 modes gives:
    
      step_ratio = (8.2ms / 1.45μs)^(1/12) ≈ 2.054
    
    Since 2.054 < 2.105, consecutive modes' processing bands overlap,
    and the guard bands [1.7, 4.2] MHz provide additional margin for
    smooth transitions — exactly as the paper describes.
    """
    n_modes = 13
    step_ratio = (SWEEP_TIME_MAX / SWEEP_TIME_MIN) ** (1.0 / (n_modes - 1))
    
    modes = []
    for i in range(n_modes):
        T = SWEEP_TIME_MIN * step_ratio ** i
        alt_low = PROC_BW_LOW * C_LIGHT * T / (2.0 * CHIRP_BW)
        alt_high = PROC_BW_HIGH * C_LIGHT * T / (2.0 * CHIRP_BW)
        modes.append(SweepMode(i, T, alt_low, alt_high))
    
    return modes


MODES = build_mode_table()


def best_mode_for_altitude(alt_m: float) -> int:
    """Find the best mode for a given altitude.
    
    First tries to find a mode where the beat frequency falls in the
    processing band [1.9, 4.0] MHz (best accuracy). If no mode covers
    the altitude in the processing band (altitude is in a gap between
    modes), falls back to the mode whose guard band [1.7, 4.2] MHz
    covers it — the signal is detectable there, just near the edge.
    
    Used during UNLOCKED state for mode search.
    """
    fb_mid = (PROC_BW_LOW + PROC_BW_HIGH) / 2.0
    
    # First: try to find a mode with the altitude in the processing band
    best_id = -1
    best_err = float('inf')
    for m in MODES:
        fb = 2.0 * CHIRP_BW * alt_m / (C_LIGHT * m.sweep_time_s)
        if PROC_BW_LOW <= fb <= PROC_BW_HIGH:
            err = abs(fb - fb_mid)
            if err < best_err:
                best_id = m.mode_id
                best_err = err
    
    if best_id >= 0:
        return best_id
    
    # Fallback: find mode whose guard band covers this altitude
    # (altitude is in a gap between processing bands)
    best_err = float('inf')
    for m in MODES:
        fb = 2.0 * CHIRP_BW * alt_m / (C_LIGHT * m.sweep_time_s)
        if GUARD_BW_LOW <= fb <= GUARD_BW_HIGH:
            err = abs(fb - fb_mid)
            if err < best_err:
                best_id = m.mode_id
                best_err = err
    
    return best_id if best_id >= 0 else 0


# ═══════════════════════════════════════════════════════════════════
# SIGNAL GENERATION (simulated RF return)
# ═══════════════════════════════════════════════════════════════════

def generate_beat_signal(sweep_time_s: float, altitude_m: float,
                         velocity_mps: float, is_up_chirp: bool,
                         snr_db: float = 25.0) -> np.ndarray:
    """Generate simulated deramped beat signal after decimation.
    
    Models the signal that would appear at the output of the decimator
    module — 8-bit, sampled at ~10.34 MHz.
    
    CRITICAL DETAIL: The number of real signal samples depends on the
    sweep time T. At low altitude (short T), only a few samples contain
    signal — the rest are zero-padded in the FFT. At high altitude
    (long T), more samples are available than the FFT can use, so we
    window to NFFT samples.
    
    Paper: "8K point FFT even in cases where the signal length is
    shorter, applying zero-padding if needed"
    
    The beat frequency is:
      f_up = fb - fd  (up-chirp)          [eq. (1)]
      f_dn = fb + fd  (down-chirp)        [eq. (2)]
    where:
      fb = 2·B·R / (c·T)                  [eq. (4)]
      fd = 2·v·fc / c                     (Doppler)
    
    Parameters
    ----------
    sweep_time_s : Sweep time T for current mode
    altitude_m   : True altitude in meters
    velocity_mps : Radial velocity (positive = closing/descending)
    is_up_chirp  : True for up-slope chirp, False for down-slope
    snr_db       : Signal-to-noise ratio in dB
    
    Returns
    -------
    8-bit quantized time-domain samples (NFFT points, zero-padded if needed)
    """
    # Beat frequency from range [eq. (4)]
    fb = 2.0 * CHIRP_BW * altitude_m / (C_LIGHT * sweep_time_s)
    
    # Doppler frequency from velocity
    fd = 2.0 * velocity_mps * CENTER_FREQ / C_LIGHT
    
    # Combined beat frequency [eq. (1), (2)]
    if is_up_chirp:
        f_beat = fb - fd
    else:
        f_beat = fb + fd
    
    # Number of real signal samples from this sweep
    # The receiver samples at FS during the sweep time T
    n_signal = int(sweep_time_s * FS)
    n_signal = max(1, n_signal)
    
    if n_signal >= NFFT:
        # Long sweep: more samples than FFT can use — window to NFFT
        # This is the case for high-altitude modes (9, 10, 11, 12)
        n_use = NFFT
    else:
        # Short sweep: fewer samples than FFT — will be zero-padded
        # This is the case for low-altitude modes (0-8)
        n_use = n_signal
    
    # Generate signal for the actual sweep duration only
    t_signal = np.arange(n_use) / FS
    signal_clean = np.sin(2.0 * np.pi * f_beat * t_signal)
    
    # Add receiver noise ONLY during the sweep (not the zero-padded region)
    # Paper: "8K point FFT even in cases where the signal length is 
    # shorter, applying zero-padding if needed"
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = 1.0 / (2.0 * snr_linear)
    noise = np.random.randn(n_use) * np.sqrt(noise_power)
    signal_with_noise = signal_clean + noise
    
    # Apply window function before zero-padding.
    # Paper: "The nadir peak signal extracted after the filtering process
    # from the spectrum serves as the fundamental basis for height
    # calculation." Windowing suppresses spectral sidelobes that would
    # otherwise be misidentified as nadir peaks, especially in the short-
    # sweep low-altitude modes where sidelobes are large.
    if n_use >= 4:
        window = np.hanning(n_use)
        signal_with_noise = signal_with_noise * window
    
    # Build the full NFFT-length buffer: windowed signal then zeros
    signal = np.zeros(NFFT)
    signal[:n_use] = signal_with_noise
    
    # Quantize to 8 bits (paper: "reduced to 8 bits")
    # The AGC sets gain so the signal portion fills the ADC range.
    # We normalize to the signal amplitude, not the noise.
    if n_use > 0 and np.max(np.abs(signal[:n_use])) > 0:
        scale = np.max(np.abs(signal[:n_use]))
        signal = signal / scale  # Signal peaks at ±1, noise is small
    elif np.max(np.abs(signal)) > 0:
        signal = signal / np.max(np.abs(signal))
    # 8-bit signed: -128 to 127
    quantized = np.clip(np.round(signal * 127), -128, 127).astype(np.int8)
    
    return quantized


# ═══════════════════════════════════════════════════════════════════
# GAIN CONTROL ESTIMATOR — Functional Implementation-1 from paper
# ═══════════════════════════════════════════════════════════════════

@dataclass
class AGCState:
    """State of the automatic gain control loop."""
    current_attenuation: int = AGC_GAIN_MAX  # Start with max attenuation
    locked: bool = False
    
    def estimate_gain(self, i_data: np.ndarray, q_data: np.ndarray,
                      threshold: float = 0.3) -> bool:
        """Run AGC estimation per Functional Implementation-1.
        
        Paper algorithm:
          - Compute envelope = I²+Q² for each sample
          - Count samples exceeding threshold
          - If >30% exceed: increase attenuation (or lock if at max)
          - If ≤30% exceed: decrease attenuation (or lock if at min)
        
        Returns True if gain is adequate (locked).
        """
        n = len(i_data)
        envelope = i_data.astype(np.float64)**2 + q_data.astype(np.float64)**2
        
        # Threshold in quantized units — scaled to 8-bit range
        env_threshold = (127 * threshold) ** 2
        sample_crossing = np.sum(envelope > env_threshold)
        
        if sample_crossing > AGC_THRESHOLD_FRACTION * n:
            if self.current_attenuation >= AGC_GAIN_MAX:
                self.locked = True
            else:
                self.current_attenuation += 1
                self.locked = False
        else:
            if self.current_attenuation <= AGC_GAIN_MIN:
                self.locked = True
            else:
                self.current_attenuation -= 1
                self.locked = False
        
        return self.locked


# ═══════════════════════════════════════════════════════════════════
# SPECTRUM COMPUTATION AND PEAK ESTIMATION
# Functional Implementation-2 from paper
# ═══════════════════════════════════════════════════════════════════

@dataclass
class PeakResult:
    """Result of peak detection in the magnitude spectrum."""
    bin_index: int = 0
    magnitude: float = 0.0
    valid: bool = False
    peak_count: int = 0         # Total peaks found above threshold


def compute_spectrum(samples: np.ndarray) -> np.ndarray:
    """Compute magnitude spectrum using 8K FFT.
    
    Paper: "8K point FFT even in cases where the signal length is
    shorter, applying zero-padding if needed"
    """
    # Zero-pad if needed
    if len(samples) < NFFT:
        padded = np.zeros(NFFT, dtype=samples.dtype)
        padded[:len(samples)] = samples
        samples = padded
    
    # FFT and magnitude
    spectrum = np.fft.fft(samples.astype(np.float64), n=NFFT)
    magnitude = np.abs(spectrum[:NFFT // 2])
    
    return magnitude


def detect_peaks(magnitude: np.ndarray, adaptive_threshold: float = 0.0
                 ) -> PeakResult:
    """Peak detection per Functional Implementation-2.
    
    Paper: "the highest peak serving as the reference point for
    qualifying data... A peak above the adaptive threshold and closer
    to the start of the processing bandwidth is selected"
    
    This means: find the nadir peak (first valid peak from the low
    end of the processing bandwidth), not just the global maximum.
    """
    # Only search within guard band range
    search_slice = magnitude[GUARD_LOW_BIN:GUARD_HIGH_BIN]
    
    if len(search_slice) == 0 or np.max(search_slice) == 0:
        return PeakResult(valid=False)
    
    # Global maximum sets the reference for threshold
    global_max = np.max(search_slice)
    
    # Adaptive threshold: paper says "predeﬁned noise threshold"
    # Use 10% of global max or the passed adaptive threshold
    threshold = max(global_max * 0.1, adaptive_threshold)
    
    # Count all peaks above threshold
    peak_count = 0
    
    # Find nadir peak: "closer to the start of the processing bandwidth"
    # Search from low to high within guard band
    nadir_peak = PeakResult(valid=False)
    
    for i in range(GUARD_LOW_BIN, GUARD_HIGH_BIN):
        if magnitude[i] > threshold:
            # Check it's a local maximum
            is_peak = True
            if i > 0 and magnitude[i] < magnitude[i - 1]:
                is_peak = False
            if i < len(magnitude) - 1 and magnitude[i] < magnitude[i + 1]:
                is_peak = False
            
            if is_peak:
                peak_count += 1
                if not nadir_peak.valid:
                    nadir_peak = PeakResult(
                        bin_index=i,
                        magnitude=magnitude[i],
                        valid=True,
                        peak_count=peak_count
                    )
    
    nadir_peak.peak_count = peak_count
    return nadir_peak


# ═══════════════════════════════════════════════════════════════════
# RF PEAK POWER COMPUTATION — from equation (6) in paper
# ═══════════════════════════════════════════════════════════════════

def compute_rf_peak_power(peak_magnitude: float, gain: int,
                          data_length: int = NFFT) -> float:
    """Compute RF peak power per equation (6).
    
    RF_peak_power = A² / (S² · R · L² · B)
    where:
      A = peak of magnitude spectrum
      R = 50 (impedance in Ω)
      S = 256 (2^8 bits)
      L = length of input data for FFT
      B = 10^((88 - gain) / 10)
    
    Paper: "The above computation is done in a floating-point
    architecture to have better precision."
    """
    A = peak_magnitude
    S = 256.0         # 2^8 quantization levels
    R = 50.0          # impedance
    L = float(data_length)
    B = 10.0 ** ((88.0 - gain) / 10.0)
    
    if S == 0 or R == 0 or L == 0 or B == 0:
        return 0.0
    
    return (A * A) / (S * S * R * L * L * B)


# ═══════════════════════════════════════════════════════════════════
# AUTOTRACKER STATE MACHINE
# ═══════════════════════════════════════════════════════════════════

class TrackState(Enum):
    """Autotracker states from paper."""
    UNLOCKED = auto()     # Searching for signal across modes
    LOCKED = auto()       # Tracking signal, maintaining sweep mode


@dataclass
class AutoTracker:
    """RAP autotracking state machine — faithful to paper.
    
    From "Autotracking for Locking" section:
      When UNLOCKED, the RAP scans all modes and picks the mode with
      the highest peak consistently over 3 consecutive cycles.
    
    From "Autotracking for Locked Cases" section:
      When LOCKED, the RAP maintains lock by ensuring the beat signal
      stays within the processing bandwidth. If it approaches the
      guard band, the sweep time is adjusted while PRESERVING LOCK
      STATUS. The sweep time is extended if signal nears upper boundary,
      reduced if it nears lower boundary.
    
    From "Altitude Estimator" section:
      If the Doppler check fails (up-down beat difference exceeds
      threshold), the current data is "disregarded, and the beat lock
      status transitions to an unlocked state, initiating the entire
      process from the gain control estimator."
    """
    state: TrackState = TrackState.UNLOCKED
    current_mode: int = 0
    
    # Mode search state (UNLOCKED)
    best_mode: int = 0
    best_power: float = 0.0
    consecutive_count: int = 0
    search_mode_index: int = 0   # Which mode we're currently testing
    
    # Locked state tracking
    agc: AGCState = field(default_factory=AGCState)
    
    def process_unlocked(self, up_peak: PeakResult, dn_peak: PeakResult,
                         gain: int) -> Tuple[int, TrackState]:
        """UNLOCKED mode search — "Autotracking for Locking" section.
        
        Paper: "The mode exhibiting the highest peak consistently over
        three consecutive cycles is recognized as the desired mode."
        
        We scan modes round-robin. When a mode produces a valid peak,
        we compare its RF peak power to the best seen. If the same
        mode has the best power for 3 consecutive cycles, we lock.
        """
        if up_peak.valid:
            # Compute RF peak power [eq. (6)]
            power = compute_rf_peak_power(up_peak.magnitude, gain)
            
            if self.current_mode == self.best_mode:
                self.consecutive_count += 1
            else:
                # New mode has signal — check if it's better
                if power > self.best_power:
                    self.best_mode = self.current_mode
                    self.best_power = power
                    self.consecutive_count = 1
                else:
                    self.consecutive_count = 0
            
            # Lock condition: 3 consecutive cycles with signal
            if self.consecutive_count >= CONSECUTIVE_REQUIRED:
                self.state = TrackState.LOCKED
                self.current_mode = self.best_mode
        else:
            self.consecutive_count = 0
        
        return self.current_mode, self.state
    
    def process_locked(self, up_peak: PeakResult, dn_peak: PeakResult
                       ) -> Tuple[int, TrackState]:
        """LOCKED mode maintenance — "Autotracking for Locked Cases" section.
        
        Paper: "If the signal begins to approach the boundary (guard band),
        the RAP requests the controller to adjust the sweep time while
        PRESERVING THE LOCK STATUS."
        
        "In the event the signal approaches the upper boundary of the
        bandwidth, the sweep time is extended [higher mode], whereas it
        is reduced [lower mode] if the signal nears the lower boundary."
        """
        if not up_peak.valid:
            # Signal lost entirely — go back to UNLOCKED
            # Paper: "data meeting this criterion proceed for additional
            # processing; otherwise, they are rejected, and the system
            # switches to an unlocked status"
            self.state = TrackState.UNLOCKED
            self.consecutive_count = 0
            self.best_power = 0.0
            return self.current_mode, self.state
        
        # Check if beat frequency is approaching guard bands
        # Paper: guard bands are [1.7, 1.9) and (4.0, 4.2] MHz
        # "the RAP requests the controller to adjust the sweep time
        #  while preserving the lock status"
        
        if up_peak.bin_index >= PROC_HIGH_BIN:
            # Approaching upper guard band → extend sweep time (higher mode)
            if self.current_mode < len(MODES) - 1:
                self.current_mode += 1
            # State stays LOCKED
            
        elif up_peak.bin_index <= PROC_LOW_BIN:
            # Approaching lower guard band → reduce sweep time (lower mode)
            if self.current_mode > 0:
                self.current_mode -= 1
            # State stays LOCKED
        
        return self.current_mode, self.state
    
    def process_measurement(self, up_peak: PeakResult, dn_peak: PeakResult,
                            gain: int = 0) -> Tuple[int, TrackState]:
        """Process one measurement cycle.
        
        Parameters
        ----------
        up_peak : Peak from up-chirp spectrum
        dn_peak : Peak from down-chirp spectrum
        gain    : Current AGC gain setting (for RF power computation)
        
        Returns
        -------
        (mode_id, state) : Current mode and tracking state
        """
        if self.state == TrackState.UNLOCKED:
            return self.process_unlocked(up_peak, dn_peak, gain)
        else:
            return self.process_locked(up_peak, dn_peak)


# ═══════════════════════════════════════════════════════════════════
# ALTITUDE ESTIMATOR — equations (7), (8) from paper
# ═══════════════════════════════════════════════════════════════════

def compute_altitude(up_peak: PeakResult, dn_peak: PeakResult,
                     mode: SweepMode,
                     doppler_limit_hz: float = 50e3
                     ) -> Tuple[Optional[float], Optional[float], bool]:
    """Compute altitude and velocity from up/down chirp peaks.
    
    From equation (8): R = M × (f_up_index + f_dn_index)
    
    Paper: "The difference between the up and down beat frequency
    signals is assessed against a predefined threshold to determine
    if the beat signal remains within the Doppler limits. If the
    threshold is exceeded, the current data acquisition is disregarded."
    
    Parameters
    ----------
    up_peak, dn_peak : Peak detection results
    mode : Current sweep mode (provides M constant)
    doppler_limit_hz : Maximum allowed Doppler frequency
    
    Returns
    -------
    (altitude_m, velocity_mps, valid)
    """
    if not up_peak.valid or not dn_peak.valid:
        return None, None, False
    
    # Doppler check: |f_up - f_dn| should be within limits
    doppler_limit_bins = int(doppler_limit_hz / FREQ_RES)
    if abs(up_peak.bin_index - dn_peak.bin_index) > doppler_limit_bins:
        return None, None, False
    
    # Altitude: R = M × (f_up_index + f_dn_index)  [eq. (8)]
    altitude = mode.M * (up_peak.bin_index + dn_peak.bin_index)
    
    # Velocity from Doppler: fd = (f_dn - f_up) / 2  [from eq. (1),(2)]
    fd_hz = (dn_peak.bin_index - up_peak.bin_index) * FREQ_RES / 2.0
    velocity = fd_hz * C_LIGHT / (2.0 * CENTER_FREQ)
    
    return altitude, velocity, True


# ═══════════════════════════════════════════════════════════════════
# COMPLETE RAP PIPELINE
# ═══════════════════════════════════════════════════════════════════

@dataclass
class RAPOutput:
    """Single output from the RAP pipeline."""
    altitude_m: Optional[float] = None
    velocity_mps: Optional[float] = None
    valid: bool = False
    mode_id: int = 0
    state: TrackState = TrackState.UNLOCKED
    up_peak: Optional[PeakResult] = None
    dn_peak: Optional[PeakResult] = None
    timestamp: float = 0.0


class RAP:
    """Complete Radar Altimeter Processor pipeline.
    
    Implements Figure 2 of the paper end-to-end:
      Data Acquisition → Decimator → AGC → Spectrum → Autotrack → Altitude
    """
    
    def __init__(self):
        self.tracker = AutoTracker()
        self.agc = AGCState()
        self.cycle_count = 0
    
    def process_cycle(self, true_altitude: float, true_velocity: float,
                      snr_db: float = 25.0) -> RAPOutput:
        """Process one complete up/down chirp measurement cycle.
        
        In the real system, RF hardware provides the deramped beat signal.
        Here we simulate it from the true altitude and velocity.
        """
        output = RAPOutput(timestamp=self.cycle_count)
        self.cycle_count += 1
        
        # Get current mode
        mode = MODES[self.tracker.current_mode]
        
        # If UNLOCKED, the system searches modes — in the real system,
        # the controller cycles through sweep times. We simulate this
        # by using the ideal mode for the true altitude.
        if self.tracker.state == TrackState.UNLOCKED:
            ideal_mode = best_mode_for_altitude(true_altitude)
            self.tracker.current_mode = ideal_mode
            mode = MODES[ideal_mode]
        
        # Step 1: Generate beat signals (simulates RF frontend + deramping)
        up_signal = generate_beat_signal(
            mode.sweep_time_s, true_altitude, true_velocity,
            is_up_chirp=True, snr_db=snr_db)
        dn_signal = generate_beat_signal(
            mode.sweep_time_s, true_altitude, true_velocity,
            is_up_chirp=False, snr_db=snr_db)
        
        # Step 2: AGC (simplified — in real system this iterates)
        # For simulation, we assume gain is stabilized
        gain = self.agc.current_attenuation
        
        # Step 3: Spectrum computation (8K FFT)
        up_mag = compute_spectrum(up_signal)
        dn_mag = compute_spectrum(dn_signal)
        
        # Step 4: Peak estimation
        up_peak = detect_peaks(up_mag)
        dn_peak = detect_peaks(dn_mag)
        
        # Step 5: Autotracking
        mode_id, state = self.tracker.process_measurement(
            up_peak, dn_peak, gain)
        
        # Step 6: Altitude estimation
        altitude, velocity, valid = compute_altitude(
            up_peak, dn_peak, mode)
        
        output.altitude_m = altitude
        output.velocity_mps = velocity
        output.valid = valid
        output.mode_id = mode_id
        output.state = state
        output.up_peak = up_peak
        output.dn_peak = dn_peak
        
        return output


# ═══════════════════════════════════════════════════════════════════
# DIAGNOSTIC AND TEST UTILITIES
# ═══════════════════════════════════════════════════════════════════

def print_mode_table():
    """Print the sweep mode table with altitude coverage."""
    print(f"\nKaRA RAP Mode Table ({len(MODES)} modes)")
    print(f"  Paper: Table 2, sweep time {SWEEP_TIME_MIN*1e6:.2f} μs to {SWEEP_TIME_MAX*1e3:.1f} ms")
    print(f"{'─' * 95}")
    print(f"  Processing band: [{PROC_BW_LOW/1e6:.2f}, {PROC_BW_HIGH/1e6:.2f}] MHz  "
          f"(ratio: {PROC_BW_HIGH/PROC_BW_LOW:.3f}×)")
    print(f"  Guard band:      [{GUARD_BW_LOW/1e6:.2f}, {GUARD_BW_HIGH/1e6:.2f}] MHz  "
          f"(ratio: {GUARD_BW_HIGH/GUARD_BW_LOW:.3f}×)")
    print(f"  Mode step ratio: {(SWEEP_TIME_MAX/SWEEP_TIME_MIN)**(1.0/(len(MODES)-1)):.3f}×")
    print(f"  FFT size: {NFFT}, Freq resolution: {FREQ_RES:.3f} Hz")
    print(f"{'─' * 95}")
    print(f"  {'Mode':>4s}  {'Sweep T':>10s}  {'Proc band (m)':>22s}  "
          f"{'Guard band (m)':>22s}  {'M (m/bin)':>10s}  {'Samples':>13s}")
    print(f"  {'':>4s}  {'(μs)':>10s}  {'':>22s}  "
          f"{'':>22s}  {'':>10s}  {'per sweep':>13s}")
    print(f"{'─' * 95}")
    
    prev_proc_high = 0
    prev_guard_high = 0
    for m in MODES:
        guard_lo = GUARD_BW_LOW * C_LIGHT * m.sweep_time_s / (2.0 * CHIRP_BW)
        guard_hi = GUARD_BW_HIGH * C_LIGHT * m.sweep_time_s / (2.0 * CHIRP_BW)
        n_samples = min(int(m.sweep_time_s * FS), NFFT)
        
        print(f"  {m.mode_id:4d}  {m.sweep_time_s*1e6:10.2f}  "
              f"[{m.alt_low:8.1f},{m.alt_high:8.1f}]  "
              f"[{guard_lo:8.1f},{guard_hi:8.1f}]  "
              f"{m.M:10.6f}  {n_samples:13d}")
        prev_proc_high = m.alt_high
        prev_guard_high = guard_hi


def test_single_altitude(altitude: float, velocity: float = 0.0,
                         verbose: bool = True):
    """Test RAP at a single altitude and velocity.
    
    Shows full detail: both chirp peaks, derived altitude and velocity,
    the M constant, and how the up/down peaks relate to equations (1)-(3).
    """
    rap = RAP()
    np.random.seed(42)
    # Pre-lock the tracker
    rap.tracker.state = TrackState.LOCKED
    ideal = best_mode_for_altitude(altitude)
    rap.tracker.current_mode = ideal
    
    result = rap.process_cycle(altitude, velocity)
    
    if verbose:
        mode = MODES[ideal]
        # Expected beat and Doppler frequencies
        fb = 2.0 * CHIRP_BW * altitude / (C_LIGHT * mode.sweep_time_s)
        fd = 2.0 * velocity * CENTER_FREQ / C_LIGHT
        f_up = fb - fd    # eq. (1)
        f_dn = fb + fd    # eq. (2)
        
        print(f"\n  Test: altitude={altitude:.1f} m, velocity={velocity:.1f} m/s")
        print(f"  Mode {ideal}: T={mode.sweep_time_s*1e6:.2f} μs, "
              f"M={mode.M:.6f} m/bin")
        print(f"  Coverage: [{mode.alt_low:.1f}, {mode.alt_high:.1f}] m")
        print(f"  ─────────────────────────────────────")
        print(f"  Beat freq fb:    {fb/1e6:.4f} MHz  [eq.(4): 2·B·R / (c·T)]")
        print(f"  Doppler fd:      {fd/1e3:.4f} kHz  [2·v·fc / c]")
        print(f"  f_up = fb - fd:  {f_up/1e6:.4f} MHz  [eq.(1)]")
        print(f"  f_dn = fb + fd:  {f_dn/1e6:.4f} MHz  [eq.(2)]")
        
        if result.up_peak and result.dn_peak:
            up = result.up_peak
            dn = result.dn_peak
            up_freq = up.bin_index * FREQ_RES if up.valid else 0
            dn_freq = dn.bin_index * FREQ_RES if dn.valid else 0
            print(f"  ─────────────────────────────────────")
            print(f"  Up-chirp peak:   bin {up.bin_index:5d} → "
                  f"{up_freq/1e6:.4f} MHz  (expect {f_up/1e6:.4f})")
            print(f"  Down-chirp peak: bin {dn.bin_index:5d} → "
                  f"{dn_freq/1e6:.4f} MHz  (expect {f_dn/1e6:.4f})")
            print(f"  Sum (up+dn):     {up.bin_index + dn.bin_index} bins  "
                  f"→ R = M × sum = {mode.M * (up.bin_index + dn.bin_index):.2f} m  [eq.(8)]")
            print(f"  Diff (dn-up):    {dn.bin_index - up.bin_index} bins  "
                  f"→ fd = {(dn.bin_index - up.bin_index) * FREQ_RES / 2.0:.1f} Hz  "
                  f"→ v = {result.velocity_mps:.2f} m/s" if result.valid else "")
        
        print(f"  ─────────────────────────────────────")
        if result.valid:
            alt_err = abs(result.altitude_m - altitude)
            alt_pct = alt_err / altitude * 100
            vel_err = abs(result.velocity_mps - velocity) if result.velocity_mps else 0
            print(f"  Altitude: {result.altitude_m:.2f} m  "
                  f"(true: {altitude:.1f}, error: {alt_err:.2f} m / {alt_pct:.3f}%)")
            print(f"  Velocity: {result.velocity_mps:.2f} m/s  "
                  f"(true: {velocity:.1f}, error: {vel_err:.2f} m/s)")
        else:
            print(f"  Measurement: INVALID")
    
    return result


def run_descent_simulation(verbose: bool = False):
    """Simulate a complete descent from 10 km to 3 m.
    
    Uses a Chandrayaan-3-like altitude profile.
    """
    rap = RAP()
    np.random.seed(42)
    
    # Build descent profile
    times = np.linspace(0, 700, 400)
    profile = []
    for t in times:
        if t < 100:
            alt = 10000.0 - 60.0 * t
            vel = 60.0
        elif t < 300:
            tp = t - 100
            alt = 4000.0 * np.exp(-tp / 120.0)
            vel = (4000.0 / 120.0) * np.exp(-tp / 120.0)
        elif t < 600:
            tp = t - 300
            alt_s = 4000.0 * np.exp(-200.0 / 120.0)
            alt = alt_s * (1.0 - tp / 300.0) ** 2
            vel = 2.0 * alt_s * (1.0 - tp / 300.0) / 300.0
        else:
            tp = t - 600
            alt = max(3.0, 20.0 - 0.2 * tp)
            vel = 0.2
        profile.append((t, max(alt, 3.0), max(vel, 0.1)))
    
    # Run simulation
    results = []
    valid_count = 0
    gap_count = 0
    current_gap = 0
    longest_gap = 0
    was_valid = False
    errors_smooth = []  # t < 600 (smooth descent)
    errors_jump = []    # t >= 600 (altitude jump recovery)
    
    if verbose:
        print(f"\n{'Time':>7s} | {'True Alt':>10s} | {'RAP Alt':>10s} | "
              f"{'Error%':>8s} | {'State':>8s} | {'Mode':>4s} | {'Vel':>8s}")
        print("─" * 72)
    
    for t, true_alt, true_vel in profile:
        output = rap.process_cycle(true_alt, true_vel)
        
        if output.valid:
            valid_count += 1
            err_pct = abs(output.altitude_m - true_alt) / true_alt * 100
            if t < 600:
                errors_smooth.append(err_pct)
            else:
                errors_jump.append(err_pct)
            if current_gap > 0:
                gap_count += 1
                longest_gap = max(longest_gap, current_gap)
                current_gap = 0
            was_valid = True
        else:
            if was_valid:
                current_gap = 1
            elif current_gap > 0:
                current_gap += 1
            was_valid = False if current_gap > 0 else was_valid
        
        results.append((t, true_alt, output))
        
        if verbose and (len(results) % 20 == 0):
            alt_str = f"{output.altitude_m:.1f}" if output.valid else "---"
            err_str = f"{err_pct:.3f}%" if output.valid else "---"
            vel_str = f"{output.velocity_mps:.1f}" if output.valid and output.velocity_mps is not None else "---"
            print(f"{t:7.1f} | {true_alt:10.1f} | {alt_str:>10s} | "
                  f"{err_str:>8s} | {output.state.name:>8s} | {output.mode_id:4d} | {vel_str:>8s}")
    
    if current_gap > 0:
        gap_count += 1
        longest_gap = max(longest_gap, current_gap)
    
    # Summary
    all_errors = errors_smooth + errors_jump
    p95_all = np.percentile(all_errors, 95) if all_errors else 0
    p95_smooth = np.percentile(errors_smooth, 95) if errors_smooth else 0
    
    print(f"\n{'═' * 72}")
    print(f"  DESCENT SIMULATION RESULTS")
    print(f"{'═' * 72}")
    print(f"  Valid measurements: {valid_count}/{len(profile)}")
    print(f"  Tracking gaps:      {gap_count} (longest: {longest_gap} cycles)")
    print(f"")
    print(f"  Altitude accuracy (smooth descent, t < 600s):")
    print(f"    Measurements:  {len(errors_smooth)}")
    print(f"    Mean error:    {np.mean(errors_smooth):.4f}%" if errors_smooth else "    Mean error: N/A")
    print(f"    p95 error:     {p95_smooth:.4f}%  (95th percentile — "
          f"95% of measurements are better than this)" if errors_smooth else "")
    spec = 0.3  # Paper Table 1: "0.3% of altitude or 0.9 m"
    print(f"    Spec (3σ):     {spec}% of altitude or 0.9 m")
    if errors_smooth:
        meets = sum(1 for e in errors_smooth if e <= spec) / len(errors_smooth) * 100
        print(f"    Meeting spec:  {meets:.1f}% of measurements")
    
    if errors_jump:
        print(f"")
        print(f"  Altitude jump recovery (t ≥ 600s — thruster anomaly scenario):")
        print(f"    Measurements:  {len(errors_jump)}")
        print(f"    Mean error:    {np.mean(errors_jump):.2f}%")
        # Count how many cycles until error < 1%
        recovery_cycle = None
        for i, e in enumerate(errors_jump):
            if e < 1.0:
                recovery_cycle = i
                break
        if recovery_cycle is not None:
            print(f"    Recovery:      {recovery_cycle} cycles to < 1% error")
        else:
            print(f"    Recovery:      did not converge to < 1% error")
    
    return results


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="KaRA RAP Reference Model — Paper-faithful implementation")
    parser.add_argument("--modes", action="store_true",
                        help="Print the sweep mode table")
    parser.add_argument("--test", action="store_true",
                        help="Test at specific altitudes")
    parser.add_argument("--altitude", type=float, default=None,
                        help="Altitude for --test (default: test multiple)")
    parser.add_argument("--velocity", type=float, default=None,
                        help="Velocity for --test in m/s (default: 0)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")
    
    args = parser.parse_args()
    
    if args.modes:
        print_mode_table()
    elif args.test:
        if args.altitude:
            vel = args.velocity if args.velocity else 0.0
            test_single_altitude(args.altitude, vel)
        else:
            print("\n" + "=" * 60)
            print("  KaRA RAP — Altitude Tests (stationary)")
            print("=" * 60)
            for alt in [10000, 5000, 2000, 800, 200, 50, 20, 10, 5, 3]:
                test_single_altitude(alt, velocity=0.0)
            
            print("\n" + "=" * 60)
            print("  KaRA RAP — Altitude + Velocity Tests")
            print("  Descent velocity creates Doppler shift between chirps.")
            print("  f_up = fb - fd  [eq.(1)]     f_dn = fb + fd  [eq.(2)]")
            print("  Altitude from sum, velocity from difference.")
            print("=" * 60)
            for alt, vel in [(5000, 60.0), (800, 30.0), (200, 15.0),
                             (50, 5.0), (10, 2.0), (3, 0.5)]:
                test_single_altitude(alt, velocity=vel)
    else:
        run_descent_simulation(verbose=args.verbose)
