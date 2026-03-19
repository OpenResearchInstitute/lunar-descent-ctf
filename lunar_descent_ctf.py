#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║   BSides San Diego 2026 — RF Village CTF                       ║
║   "LUNAR DESCENT"                                              ║
║                                                                ║
║   The radar altimeter works. The autopilot crashes.            ║
║                                                                ║
║   ISRO's KaRA radar altimeter guided Chandrayaan-3 to a safe  ║
║   lunar landing. The RAP (Radar Altimeter Processor) computes  ║
║   altitude and velocity from FMCW chirp signals.              ║
║                                                                ║
║   The altimeter measurements are fed to a landing autopilot.  ║
║   During testing, the autopilot crashes the lander every time ║
║   below 15 meters altitude. The altitude readings look fine.  ║
║   Something else is wrong.                                     ║
║                                                                ║
║   Your job: find why the autopilot crashes, fix the sensor    ║
║   qualification logic, and land the spacecraft safely.         ║
║                                                                ║
║   Three flags. Zero are free.                                  ║
║   Submit flags at the RF Village table.                        ║
║                                                                ║
║   Open Research Institute — openresearch.institute              ║
║   Based on Sharma et al., IEEE A&E Systems Mag, Jan 2026      ║
╚══════════════════════════════════════════════════════════════════╝

SETUP:
    pip install numpy matplotlib
    python lunar_descent_ctf.py --help            # See all options
    python lunar_descent_ctf.py                   # Run the mission
    python lunar_descent_ctf.py --test -p all     # Test all profiles
    python lunar_descent_ctf.py --score           # Check your flags

THE PHYSICS:
    The RAP uses FMCW (frequency-modulated continuous-wave) radar.
    Up-chirp and down-chirp signals are transmitted and received.

    Beat frequencies:
      f_up = fb - fd    (up-chirp)     [eq. 1]
      f_dn = fb + fd    (down-chirp)   [eq. 2]

    Altitude comes from the SUM of the peaks:
      R = M × (f_up_index + f_dn_index)          [eq. 8]

    Velocity comes from the DIFFERENCE:
      fd = (f_dn_index - f_up_index) × freq_res / 2

    The sweep time T determines which altitude range maps into
    the processing bandwidth [1.9, 4.0] MHz. Short T = low altitude,
    long T = high altitude. The 8K FFT is always the same size,
    but the number of signal samples depends on T.

    At low altitude (mode 0, T=1.45μs): only 14 signal samples.
    At high altitude (mode 9+, T>945μs): full 8192 samples.

FLAGS:
    1. RECON (100 pts)          — Explain the bug to village staff
    2. FIRST LIGHT (500 pts)    — Land all three profiles
    3. NO GAPS (400 pts)        — Zero tracking gaps on all profiles

RULES:
    - Edit ONLY the MeasurementQualifier class (marked below)
    - Don't change the RAP, signal generation, or autopilot
    - The qualifier decides what the autopilot sees
    - Flags are earned by landing performance, not hacking the scorer

Good luck. Don't crash. 🌙
"""

import numpy as np
import struct
import hashlib
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Tuple, Optional, List


# ═══════════════════════════════════════════════════════════════════
# SYSTEM PARAMETERS — from Sharma et al., Table 1 & 2
# ═══════════════════════════════════════════════════════════════════

CENTER_FREQ = 35.61e9          # Hz — Ka-band
CHIRP_BW = 240e6               # Hz — transmitted bandwidth
C_LIGHT = 299_792_458.0        # m/s

ADC_RATE = 496.53e6            # Hz
DECIMATION_FACTOR = 48
FS = ADC_RATE / DECIMATION_FACTOR  # ~10.34 MHz
ADC_BITS = 8

NFFT = 8192                    # 8K FFT
FREQ_RES = FS / NFFT           # ~1.263 kHz/bin

PROC_BW_LOW = 1.9e6            # Hz
PROC_BW_HIGH = 4.0e6           # Hz
GUARD_BW_LOW = 1.7e6           # Hz
GUARD_BW_HIGH = 4.2e6          # Hz

PROC_LOW_BIN = int(PROC_BW_LOW / FREQ_RES)
PROC_HIGH_BIN = int(PROC_BW_HIGH / FREQ_RES)
GUARD_LOW_BIN = int(GUARD_BW_LOW / FREQ_RES)
GUARD_HIGH_BIN = int(GUARD_BW_HIGH / FREQ_RES)

SWEEP_TIME_MIN = 1.45e-6       # s — paper Table 2
SWEEP_TIME_MAX = 8.2e-3        # s
CONSECUTIVE_REQUIRED = 3


# ═══════════════════════════════════════════════════════════════════
# SWEEP MODES — 13 modes, paper endpoints 1.45μs to 8.2ms
# ═══════════════════════════════════════════════════════════════════

@dataclass
class SweepMode:
    mode_id: int
    sweep_time_s: float
    alt_low: float = 0.0
    alt_high: float = 0.0

    @property
    def M(self) -> float:
        """R = M × (f_up_index + f_dn_index), IEEE-754 single precision."""
        val = (C_LIGHT * self.sweep_time_s * FS) / (4.0 * CHIRP_BW * NFFT)
        return struct.unpack('f', struct.pack('f', val))[0]

    @property
    def samples_per_sweep(self) -> int:
        """Number of real signal samples in one sweep."""
        return min(int(self.sweep_time_s * FS), NFFT)


def _build_modes():
    n_modes = 13
    step = (SWEEP_TIME_MAX / SWEEP_TIME_MIN) ** (1.0 / (n_modes - 1))
    modes = []
    for i in range(n_modes):
        T = SWEEP_TIME_MIN * step ** i
        lo = PROC_BW_LOW * C_LIGHT * T / (2.0 * CHIRP_BW)
        hi = PROC_BW_HIGH * C_LIGHT * T / (2.0 * CHIRP_BW)
        modes.append(SweepMode(i, T, lo, hi))
    return modes

MODES = _build_modes()


def best_mode_for_altitude(alt_m):
    fb_mid = (PROC_BW_LOW + PROC_BW_HIGH) / 2.0
    best_id, best_err = -1, float('inf')
    for m in MODES:
        fb = 2.0 * CHIRP_BW * alt_m / (C_LIGHT * m.sweep_time_s)
        if PROC_BW_LOW <= fb <= PROC_BW_HIGH:
            err = abs(fb - fb_mid)
            if err < best_err:
                best_id, best_err = m.mode_id, err
    if best_id >= 0:
        return best_id
    for m in MODES:
        fb = 2.0 * CHIRP_BW * alt_m / (C_LIGHT * m.sweep_time_s)
        if GUARD_BW_LOW <= fb <= GUARD_BW_HIGH:
            err = abs(fb - fb_mid)
            if err < best_err:
                best_id, best_err = m.mode_id, err
    return best_id if best_id >= 0 else 0


# ═══════════════════════════════════════════════════════════════════
# SIGNAL GENERATION — DO NOT MODIFY
# ═══════════════════════════════════════════════════════════════════

@dataclass
class PeakResult:
    bin_index: int = 0
    magnitude: float = 0.0
    valid: bool = False


def generate_beat_signal(sweep_time_s, altitude_m, velocity_mps,
                         is_up_chirp, snr_db=25.0):
    fb = 2.0 * CHIRP_BW * altitude_m / (C_LIGHT * sweep_time_s)
    fd = 2.0 * velocity_mps * CENTER_FREQ / C_LIGHT
    f_beat = fb - fd if is_up_chirp else fb + fd

    n_use = min(max(1, int(sweep_time_s * FS)), NFFT)
    t_sig = np.arange(n_use) / FS
    sig_clean = np.sin(2.0 * np.pi * f_beat * t_sig)

    snr_lin = 10.0 ** (snr_db / 10.0)
    noise = np.random.randn(n_use) * np.sqrt(1.0 / (2.0 * snr_lin))
    sig_noisy = sig_clean + noise

    if n_use >= 4:
        sig_noisy = sig_noisy * np.hanning(n_use)

    signal = np.zeros(NFFT)
    signal[:n_use] = sig_noisy

    if n_use > 0 and np.max(np.abs(signal[:n_use])) > 0:
        signal = signal / np.max(np.abs(signal[:n_use]))
    quantized = np.clip(np.round(signal * 127), -128, 127).astype(np.int8)
    return quantized


def compute_spectrum(samples):
    if len(samples) < NFFT:
        padded = np.zeros(NFFT, dtype=samples.dtype)
        padded[:len(samples)] = samples
        samples = padded
    spectrum = np.fft.fft(samples.astype(np.float64), n=NFFT)
    return np.abs(spectrum[:NFFT // 2])


def detect_peak(magnitude):
    search = magnitude[GUARD_LOW_BIN:GUARD_HIGH_BIN]
    if len(search) == 0 or np.max(search) == 0:
        return PeakResult(valid=False)
    threshold = np.max(search) * 0.1
    for i in range(GUARD_LOW_BIN, GUARD_HIGH_BIN):
        if magnitude[i] > threshold:
            is_pk = True
            if i > 0 and magnitude[i] < magnitude[i - 1]:
                is_pk = False
            if i < len(magnitude) - 1 and magnitude[i] < magnitude[i + 1]:
                is_pk = False
            if is_pk:
                return PeakResult(bin_index=i, magnitude=magnitude[i], valid=True)
    return PeakResult(valid=False)


# ═══════════════════════════════════════════════════════════════════
# AUTOTRACKER — paper-faithful, DO NOT MODIFY
# ═══════════════════════════════════════════════════════════════════

class TrackState(Enum):
    UNLOCKED = auto()
    LOCKED = auto()


class AutoTracker:
    def __init__(self):
        self.state = TrackState.UNLOCKED
        self.current_mode = 0
        self.consecutive = 0
        self.best_mode = 0
        self.best_power = 0.0

    def process_measurement(self, up_peak, dn_peak, gain=0):
        if self.state == TrackState.UNLOCKED:
            if up_peak.valid:
                if self.best_mode == self.current_mode:
                    self.consecutive += 1
                else:
                    self.best_mode = self.current_mode
                    self.best_power = up_peak.magnitude
                    self.consecutive = 1
                if self.consecutive >= CONSECUTIVE_REQUIRED:
                    self.state = TrackState.LOCKED
                    self.current_mode = self.best_mode
            else:
                self.consecutive = 0

        elif self.state == TrackState.LOCKED:
            if not up_peak.valid:
                self.state = TrackState.UNLOCKED
                self.consecutive = 0
                self.best_power = 0.0
                return self.current_mode, self.state

            if up_peak.bin_index >= PROC_HIGH_BIN:
                if self.current_mode < len(MODES) - 1:
                    self.current_mode += 1
            elif up_peak.bin_index <= PROC_LOW_BIN:
                if self.current_mode > 0:
                    self.current_mode -= 1

        return self.current_mode, self.state


# ═══════════════════════════════════════════════════════════════════
# ALTITUDE & VELOCITY COMPUTATION — DO NOT MODIFY
# ═══════════════════════════════════════════════════════════════════

def compute_altitude_velocity(up_peak, dn_peak, mode):
    """Compute altitude [eq.8] and velocity from Doppler.
    Returns (altitude, velocity, valid).
    """
    if not up_peak.valid or not dn_peak.valid:
        return None, None, False

    altitude = mode.M * (up_peak.bin_index + dn_peak.bin_index)
    fd_hz = (dn_peak.bin_index - up_peak.bin_index) * FREQ_RES / 2.0
    velocity = fd_hz * C_LIGHT / (2.0 * CENTER_FREQ)

    return altitude, velocity, True


# ═══════════════════════════════════════════════════════════════════
# RAP PIPELINE — DO NOT MODIFY
# ═══════════════════════════════════════════════════════════════════

@dataclass
class RAPOutput:
    altitude_m: Optional[float] = None
    velocity_mps: Optional[float] = None
    valid: bool = False
    mode_id: int = 0
    state: TrackState = TrackState.UNLOCKED
    up_peak: Optional[PeakResult] = None
    dn_peak: Optional[PeakResult] = None


class RAP:
    def __init__(self):
        self.tracker = AutoTracker()

    def process_cycle(self, true_altitude, true_velocity, snr_db=25.0):
        output = RAPOutput()
        mode = MODES[self.tracker.current_mode]

        if self.tracker.state == TrackState.UNLOCKED:
            ideal = best_mode_for_altitude(true_altitude)
            self.tracker.current_mode = ideal
            mode = MODES[ideal]

        up_sig = generate_beat_signal(mode.sweep_time_s, true_altitude,
                                       true_velocity, True, snr_db)
        dn_sig = generate_beat_signal(mode.sweep_time_s, true_altitude,
                                       true_velocity, False, snr_db)

        up_peak = detect_peak(compute_spectrum(up_sig))
        dn_peak = detect_peak(compute_spectrum(dn_sig))

        mode_id, state = self.tracker.process_measurement(up_peak, dn_peak)
        altitude, velocity, valid = compute_altitude_velocity(up_peak, dn_peak, mode)

        output.altitude_m = altitude
        output.velocity_mps = velocity
        output.valid = valid
        output.mode_id = mode_id
        output.state = state
        output.up_peak = up_peak
        output.dn_peak = dn_peak
        return output


# ═══════════════════════════════════════════════════════════════════
#
#   ██████╗ ████████╗███████╗    ████████╗ █████╗ ██████╗  ██████╗ ███████╗████████╗
#  ██╔════╝ ╚══██╔══╝██╔════╝    ╚══██╔══╝██╔══██╗██╔══██╗██╔════╝ ██╔════╝╚══██╔══╝
#  ██║         ██║   █████╗         ██║   ███████║██████╔╝██║  ███╗█████╗     ██║
#  ██║         ██║   ██╔══╝         ██║   ██╔══██║██╔══██╗██║   ██║██╔══╝     ██║
#  ╚██████╗    ██║   ██║            ██║   ██║  ██║██║  ██║╚██████╔╝███████╗   ██║
#   ╚═════╝    ╚═╝   ╚═╝            ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝   ╚═╝
#
#   EDIT THIS CLASS TO FIX THE BUG
#   Everything else is off-limits.
#
# ═══════════════════════════════════════════════════════════════════

class MeasurementQualifier:
    """Measurement qualification logic for the landing autopilot.

    The RAP computes altitude and velocity every cycle. This qualifier
    decides what the autopilot actually USES. If a measurement is bad,
    the qualifier should reject it or substitute a safe value so the
    autopilot doesn't act on garbage.

    The paper mentions "three sample qualification logics to generate
    the final altitude, thereby enhancing the robustness of the design"
    but doesn't detail them. This is where those logics live.

    Currently, this qualifier applies a fixed Doppler limit check
    (from the paper's altitude estimator section) and passes everything
    else through to the autopilot. It was validated during field tests
    on helicopters at altitudes above 50m. Nobody tested below 30m.

    YOUR JOB: Figure out why the autopilot crashes below 15m and fix
    the qualification logic. You may:
    - Add new fields and methods
    - Change the qualify() method
    - Add per-mode logic
    - Use any information from the RAPOutput

    You may NOT change the method signature of qualify().
    """

    def __init__(self):
        # Fixed Doppler limit — same for all modes
        # Paper: "predefined threshold"
        self.doppler_limit_bins = 40   # ~50 kHz

        # History for smoothing (you can use these)
        self.last_good_altitude = None
        self.last_good_velocity = None

    def qualify(self, rap_output: RAPOutput) -> Tuple[Optional[float],
                                                       Optional[float], bool]:
        """Qualify a RAP measurement for the autopilot.

        Parameters
        ----------
        rap_output : RAPOutput from the RAP pipeline

        Returns
        -------
        (altitude_m, velocity_mps, valid)
            What the autopilot will use. If valid=False, the autopilot
            holds its last command (coasts).
        """
        if not rap_output.valid:
            return None, None, False

        # Doppler sanity check — reject if up/down peaks are too far apart
        if rap_output.up_peak and rap_output.dn_peak:
            doppler_bins = abs(rap_output.up_peak.bin_index -
                              rap_output.dn_peak.bin_index)
            if doppler_bins > self.doppler_limit_bins:
                return None, None, False

        # *** THE BUG IS HERE ***
        # Everything that passes the Doppler check goes straight to
        # the autopilot — altitude AND velocity. The altitude is fine
        # at all modes (zero-padding + windowing works). But what about
        # the velocity at low-altitude modes where the FFT only has
        # 14-63 signal samples? The Doppler (velocity) is derived from
        # the DIFFERENCE of two peak positions. With so few samples,
        # the peak positions have noise of several bins — which maps
        # to velocity errors of ±30 m/s or worse.
        #
        # The autopilot TRUSTS this velocity and fires thrusters to
        # "correct" for motion that doesn't exist.

        self.last_good_altitude = rap_output.altitude_m
        self.last_good_velocity = rap_output.velocity_mps

        return rap_output.altitude_m, rap_output.velocity_mps, True


# ═══════════════════════════════════════════════════════════════════
#  END OF EDITABLE SECTION
# ═══════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════
# LANDING AUTOPILOT — DO NOT MODIFY
# ═══════════════════════════════════════════════════════════════════

@dataclass
class AutopilotState:
    """Simplified landing autopilot state."""
    position_m: float = 0.0       # Estimated altitude from qualified data
    velocity_mps: float = 0.0     # Estimated velocity from qualified data
    thrust_cmd: float = 0.0       # Current thrust command
    max_position_error_m: float = 0.0
    max_velocity_error: float = 0.0
    crashed: bool = False
    coast_cycles: int = 0

    # Tuning
    VELOCITY_GAIN: float = 2.0
    MAX_THRUST: float = 50.0
    # Crash condition: autopilot is below this altitude AND sees
    # velocity exceeding the threshold. At high altitude, large velocity
    # is expected (rapid descent). Near the ground, it means sensor noise
    # is feeding garbage into thrust control.
    CRASH_ALT_M: float = 30.0     # Below this, velocity must be sane
    CRASH_VELOCITY: float = 20.0  # Max believable velocity near ground


class LandingAutopilot:
    """Simplified landing autopilot that uses qualified RAP measurements.

    Uses velocity feedback to control descent rate. If the velocity
    reading is wrong near the ground, it applies wrong thrust.

    CRASH CONDITION: If the autopilot is below 30m altitude and
    receives a velocity reading exceeding 20 m/s, it fires full
    thrust in the wrong direction — crash.

    At high altitude, large velocities are expected and don't
    trigger a crash. The problem only manifests during final approach.
    """

    def __init__(self):
        self.state = AutopilotState()

    def update(self, altitude_m, velocity_mps, valid, true_altitude):
        if valid and altitude_m is not None and velocity_mps is not None:
            self.state.position_m = altitude_m
            self.state.velocity_mps = velocity_mps
            self.state.coast_cycles = 0

            # Track position error
            self.state.max_position_error_m = max(
                self.state.max_position_error_m,
                abs(altitude_m - true_altitude))

            # Track max velocity magnitude
            self.state.max_velocity_error = max(
                self.state.max_velocity_error,
                abs(velocity_mps))

            # Crash check: near the ground with absurd velocity
            # This is the scenario: autopilot thinks it's at 5m and
            # moving at 40 m/s sideways. It fires full lateral thrust.
            # The lander tips over and crashes.
            if (altitude_m < self.state.CRASH_ALT_M and
                    abs(velocity_mps) > self.state.CRASH_VELOCITY):
                self.state.crashed = True

        else:
            self.state.coast_cycles += 1


# ═══════════════════════════════════════════════════════════════════
# MISSION PROFILES — DO NOT MODIFY
# ═══════════════════════════════════════════════════════════════════

def build_profile(name):
    """Build a descent profile. Returns list of (time, altitude, velocity)."""
    if name == "standard":
        # Chandrayaan-3-like smooth descent + altitude excursion
        # The altitude jump at t=600 simulates a thruster anomaly
        # during hazard avoidance — altitude suddenly increases.
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

    elif name == "aggressive":
        # Fast exponential descent — rapid mode transitions at top,
        # velocity noise at bottom.
        times = np.linspace(0, 400, 400)
        profile = []
        for t in times:
            if t < 350:
                alt = 10000.0 * np.exp(-t / 60.0)
                vel = (10000.0 / 60.0) * np.exp(-t / 60.0)
            else:
                tp = t - 350
                alt_350 = 10000.0 * np.exp(-350 / 60.0)
                alt = max(3.0, alt_350 * (1.0 - tp / 50.0))
                vel = max(0.1, alt_350 / 50.0)
            profile.append((t, max(alt, 3.0), max(vel, 0.1)))

    elif name == "stepwise":
        # Hover at guard band boundaries, drop between them.
        # Forces mode transitions at every step, ends with sustained
        # hover at 10m where velocity is unreliable.
        hover_alts = [9851.0, 4795.0, 2334.0, 553.0, 131.0, 31.0, 5.0]
        times = np.linspace(0, 600, 400)
        seg_dur = 600.0 / len(hover_alts)
        drop_frac = 0.15  # 15% of segment is descent

        profile = []
        for t in times:
            seg = min(int(t / seg_dur), len(hover_alts) - 1)
            seg_t = t - seg * seg_dur
            prev_alt = hover_alts[seg - 1] if seg > 0 else 10000.0
            target = hover_alts[seg]

            if seg > 0 and seg_t < seg_dur * drop_frac:
                frac = seg_t / (seg_dur * drop_frac)
                alt = prev_alt + (target - prev_alt) * frac
                vel = abs(target - prev_alt) / (seg_dur * drop_frac)
            else:
                alt = target
                vel = 0.1
            profile.append((t, max(alt, 3.0), max(vel, 0.1)))

    else:
        raise ValueError(f"Unknown profile: {name}")

    return profile


# ═══════════════════════════════════════════════════════════════════
# MISSION SIMULATION
# ═══════════════════════════════════════════════════════════════════

@dataclass
class MissionResult:
    profile_name: str = ""
    total_measurements: int = 0
    valid_measurements: int = 0
    tracking_gaps: int = 0
    longest_gap: int = 0
    max_position_error_m: float = 0.0
    max_velocity_error_mps: float = 0.0
    qualifier_rejections: int = 0  # RAP said valid, qualifier said no
    rap_invalid: int = 0           # RAP itself couldn't produce data
    crashed: bool = False
    landed: bool = False


def run_mission(profile_name, seed=42, verbose=False):
    """Run one descent profile through RAP + qualifier + autopilot."""
    np.random.seed(seed)
    rap = RAP()
    qualifier = MeasurementQualifier()
    autopilot = LandingAutopilot()
    profile = build_profile(profile_name)

    results = []
    gap_count = 0
    current_gap = 0
    longest_gap = 0
    was_valid = False
    qualifier_rejections = 0
    rap_invalid = 0

    if verbose:
        print(f"\n{'Time':>7s} | {'True Alt':>9s} | {'RAP Alt':>9s} | "
              f"{'RAP Vel':>8s} | {'Q.Vel':>7s} | {'Mode':>4s} | {'Status':>8s}")
        print("─" * 68)

    for t, true_alt, true_vel in profile:
        rap_out = rap.process_cycle(true_alt, true_vel)
        q_alt, q_vel, q_valid = qualifier.qualify(rap_out)
        autopilot.update(q_alt, q_vel, q_valid, true_alt)

        # Track qualifier rejections vs RAP invalids
        if not rap_out.valid:
            rap_invalid += 1
        elif not q_valid:
            qualifier_rejections += 1

        # Track gaps in qualified output
        if q_valid:
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

        results.append((t, true_alt, true_vel, rap_out, q_alt, q_vel, q_valid))

        if verbose and (len(results) % 20 == 0):
            r_alt = f"{rap_out.altitude_m:.1f}" if rap_out.valid else "---"
            r_vel = f"{rap_out.velocity_mps:.1f}" if rap_out.valid and rap_out.velocity_mps is not None else "---"
            qv = f"{q_vel:.1f}" if q_valid and q_vel is not None else "---"
            status = "CRASH" if autopilot.state.crashed else "OK"
            print(f"{t:7.1f} | {true_alt:9.1f} | {r_alt:>9s} | "
                  f"{r_vel:>8s} | {qv:>7s} | {rap_out.mode_id:4d} | {status:>8s}")

    if current_gap > 0:
        gap_count += 1
        longest_gap = max(longest_gap, current_gap)

    mr = MissionResult()
    mr.profile_name = profile_name
    mr.total_measurements = len(profile)
    mr.valid_measurements = sum(1 for (_, _, _, _, _, _, v) in results if v)
    mr.tracking_gaps = gap_count
    mr.longest_gap = longest_gap
    mr.max_position_error_m = autopilot.state.max_position_error_m
    mr.max_velocity_error_mps = autopilot.state.max_velocity_error
    mr.qualifier_rejections = qualifier_rejections
    mr.rap_invalid = rap_invalid
    mr.crashed = autopilot.state.crashed
    mr.landed = (not autopilot.state.crashed and
                 any(r[4] is not None and r[4] < 15.0 for r in results))

    return mr, results


# ═══════════════════════════════════════════════════════════════════
# SCORING AND FLAGS — DO NOT MODIFY
# ═══════════════════════════════════════════════════════════════════

def _flag(seed_str):
    h = hashlib.sha256(f"ORI-LUNAR-{seed_str}-2026".encode()).hexdigest()[:12]
    return f"FLAG{{lunar_{h}}}"


def score_and_flags():
    """Run all profiles and compute flags.
    
    Challenge 1: Velocity Qualification
      Flag 1 (100 pts) RECON:       Explain the bug to staff.
      Flag 2 (500 pts) FIRST LIGHT: Land all three profiles.
    
    Challenge 2: Full Coverage
      Flag 3 (400 pts) NO GAPS:     Zero tracking gaps on all profiles.
    
    Buggy code scores: 0 / 1000.
    """
    print("\n" + "=" * 68)
    print("  LUNAR DESCENT CTF — SCORING")
    print("=" * 68)

    profiles = ["standard", "aggressive", "stepwise"]
    all_results = {}

    for pname in profiles:
        mr, _ = run_mission(pname, seed=42)
        all_results[pname] = mr
        status = "LANDED" if mr.landed else "CRASHED" if mr.crashed else "LOST"
        print(f"\n  Profile: {pname}")
        print(f"    Valid:          {mr.valid_measurements}/{mr.total_measurements}")
        print(f"    Qualifier rejected: {mr.qualifier_rejections} "
              f"(RAP had data, qualifier dropped it)")
        print(f"    RAP invalid:    {mr.rap_invalid} (no signal)")
        print(f"    Position error: {mr.max_position_error_m:.2f} m")
        print(f"    Max velocity:   {mr.max_velocity_error_mps:.1f} m/s")
        print(f"    Status:         {status}")

    total_score = 0
    flags_earned = []

    # ── FLAG 1: RECON (100 pts) ─── staff validated ──────────
    print(f"\n{'─' * 68}")
    print(f"  FLAG 1 — RECON (100 pts)")
    print(f"  The autopilot crashes the lander.")
    print(f"  Explain to RF Village staff what is the root cause?")
    print(f"  [Flag issued by staff — no hash here]")

    # ── FLAG 2: FIRST LIGHT (500 pts) ── land all ────────────
    all_landed = all(all_results[p].landed for p in profiles)
    print(f"\n  FLAG 2 — FIRST LIGHT (500 pts)")
    if all_landed:
        flag2 = _flag("first-light-vel-qual")
        print(f"  All three landers survived!")
        print(f"  {flag2}")
        flags_earned.append(("FIRST LIGHT", 500, flag2))
        total_score += 500
    else:
        print(f"  LOCKED: Land ALL three profiles without crashing")
        for p in profiles:
            mr = all_results[p]
            status = "LANDED" if mr.landed else "CRASHED" if mr.crashed else "LOST"
            extra = ""
            if mr.crashed:
                extra = f" (vel={mr.max_velocity_error_mps:.0f} m/s)"
            print(f"    {p}: {status}{extra}")

    # ── FLAG 3: NO GAPS (400 pts) ── full coverage ─────────────
    # The simplistic Flag 2 fix (reject velocity at low modes) still
    # leaves rejected cycles — the fixed Doppler threshold throws away
    # valid high-velocity measurements during fast descent. The crew
    # is flying blind at high speed. Fix the Doppler qualification to
    # accept real high-velocity data while still rejecting noise.
    #
    # We count QUALIFIER rejections, not RAP invalids. The RAP has
    # a few genuine signal losses (mode transitions, altitude jump)
    # that no qualifier can fix. What matters is: when the RAP gives
    # you good data, does the qualifier pass it through?
    total_rejections = sum(all_results[p].qualifier_rejections for p in profiles)
    all_no_rejections = total_rejections == 0 and all_landed
    print(f"\n  FLAG 3 — NO GAPS (400 pts)")
    if all_no_rejections:
        flag3 = _flag("no-gaps-full-coverage")
        print(f"  Zero qualifier rejections!")
        print(f"  Every valid RAP measurement reaches the autopilot.")
        print(f"  {flag3}")
        flags_earned.append(("NO GAPS", 400, flag3))
        total_score += 400
    else:
        print(f"  LOCKED: Zero qualifier rejections on ALL profiles")
        print(f"  (The qualifier must pass through ALL valid RAP data)")
        for p in profiles:
            mr = all_results[p]
            if not mr.landed:
                status = "CRASHED" if mr.crashed else "LOST"
                print(f"    {p}: {status}")
            elif mr.qualifier_rejections == 0:
                print(f"    {p}: CLEAN (0 rejections)")
            else:
                print(f"    {p}: {mr.qualifier_rejections} valid measurements rejected")

    # Summary
    print(f"\n{'═' * 68}")
    print(f"  TOTAL SCORE: {total_score} / 1000 points")
    print(f"  FLAGS: {len(flags_earned)} / 3")
    print(f"  (Flag 1 is staff-validated at the RF Village table)")
    print(f"{'═' * 68}")

    if total_score >= 900:
        print(f"\n  🌙 Full coverage, all landers safe.")
        print(f"  Show this to the RF Village table for your prize.")
    elif total_score >= 500:
        print(f"\n  Landers survived, but there are tracking gaps.")
        print(f"  The crew is flying blind during fast descent.")
        print(f"  Look at the X marks in the timeline — why are valid")
        print(f"  measurements being rejected?")
    else:
        print(f"\n  Score: 0. All three landers crashed on autopilot.")
        print(f"  The altitude readings are fine. Look at the velocity.")
        print(f"  How many signal samples does mode 0 have? (try --modes)")

    return total_score, flags_earned


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

VALID_PROFILES = ["standard", "aggressive", "stepwise", "all"]


def run_test(profile_names, verbose=True):
    profiles = profile_names if isinstance(profile_names, list) else [profile_names]
    for pname in profiles:
        mr, results = run_mission(pname, seed=42, verbose=verbose)

        # Timeline
        print(f"\n  Tracking Timeline:")
        print(f"  (. = OK, X = gap, ! = bad velocity near ground, altitude decreasing →)\n")
        target_width = 72
        window = max(1, len(results) // target_width)
        line = "  "
        for i in range(0, len(results), window):
            chunk = results[i:i + window]
            # Check for dangerous velocity: near ground + high velocity
            bad_vel = any(
                r[5] is not None and r[4] is not None and
                r[4] < AutopilotState.CRASH_ALT_M and abs(r[5]) > AutopilotState.CRASH_VELOCITY
                for r in chunk if r[6]  # r[5]=q_vel, r[4]=q_alt, r[6]=q_valid
            )
            has_gap = not all(r[6] for r in chunk)
            if bad_vel:
                line += "!"
            elif has_gap:
                line += "X"
            else:
                line += "."
        print(line)

        status = "LANDED" if mr.landed else "CRASHED" if mr.crashed else "LOST"
        print(f"\n  {pname}: {status}")
        print(f"    Valid: {mr.valid_measurements}/{mr.total_measurements}")
        print(f"    Gaps: {mr.tracking_gaps}, Position error: {mr.max_position_error_m:.2f}m")
        print(f"    Max velocity seen: {mr.max_velocity_error_mps:.1f} m/s")
        if mr.crashed:
            print(f"\n  💥 The autopilot crashed the lander.")
            print(f"  It received a velocity reading that exceeded {AutopilotState.CRASH_VELOCITY} m/s.")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        prog="lunar_descent_ctf",
        description="""
╔══════════════════════════════════════════════════════════════╗
║  LUNAR DESCENT — BSides San Diego 2026 RF Village CTF      ║
║                                                            ║
║  The radar altimeter works. The autopilot crashes.         ║
║  The altitude is fine. The velocity is not.                ║
║                                                            ║
║  Fix the MeasurementQualifier class.                       ║
║  3 flags, 1000 points. Zero are free.                      ║
║                                                            ║
║  Open Research Institute — openresearch.institute           ║
╚══════════════════════════════════════════════════════════════╝
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  %(prog)s                        Run the default (standard profile)
  %(prog)s --test -p stepwise     Test the stepwise profile
  %(prog)s --test -p all          Test all three profiles
  %(prog)s --score                Score your fix and earn flags

profiles:
  standard    Smooth descent + altitude excursion at 20m
  aggressive  Fast exponential braking, rapid mode transitions
  stepwise    Hover at mode boundaries, final hover at 5m

flags:
  1. RECON (100 pts)           Explain the bug to village staff
  2. FIRST LIGHT (500 pts)     Land all three profiles
  3. NO GAPS (400 pts)         Zero tracking gaps on all profiles
""",
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--test", action="store_true",
                      help="Run verbose test")
    mode.add_argument("--score", action="store_true",
                      help="Score your fix and earn flags")
    mode.add_argument("--modes", action="store_true",
                      help="Print the sweep mode table")

    parser.add_argument("-p", "--profile", choices=VALID_PROFILES,
                        default="standard",
                        help="Profile to test (default: standard)")

    args = parser.parse_args()

    if args.score:
        score_and_flags()
    elif args.modes:
        print(f"\nKaRA RAP Mode Table ({len(MODES)} modes)")
        print(f"  Paper: Table 2, sweep time {SWEEP_TIME_MIN*1e6:.2f} μs to {SWEEP_TIME_MAX*1e3:.1f} ms")
        print(f"  Processing band: [{PROC_BW_LOW/1e6:.2f}, {PROC_BW_HIGH/1e6:.2f}] MHz")
        print(f"  FFT size: {NFFT}, Freq resolution: {FREQ_RES:.1f} Hz")
        print(f"{'─' * 78}")
        print(f"  {'Mode':>4s}  {'Sweep T (μs)':>12s}  {'Alt range (m)':>22s}  "
              f"{'M (m/bin)':>10s}  {'Samples':>12s}")
        print(f"{'─' * 78}")
        for m in MODES:
            n = m.samples_per_sweep
            print(f"  {m.mode_id:4d}  {m.sweep_time_s*1e6:12.2f}  "
                  f"[{m.alt_low:8.1f},{m.alt_high:8.1f}]  "
                  f"{m.M:10.6f}  {n:8d}/{NFFT}")
    elif args.test:
        if args.profile == "all":
            run_test(["standard", "aggressive", "stepwise"])
        else:
            run_test(args.profile)
    else:
        if args.profile == "all":
            run_test(["standard", "aggressive", "stepwise"])
        else:
            run_test(args.profile)


if __name__ == "__main__":
    main()
