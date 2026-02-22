"""
╔══════════════════════════════════════════════════════════════════╗
║   BSides San Diego 2026 — RF Village CTF                       ║
║   "LUNAR DESCENT"                                              ║
║                                                                ║
║   Can you land the spacecraft?                                 ║
║                                                                ║
║   The radar altimeter tracks altitude during descent, but      ║
║   something goes wrong during mode transitions. The            ║
║   autotracker loses lock and the altitude readings go dark.    ║
║                                                                ║
║   Your job: find the bug, explain it, fix it, land safely.     ║
║                                                                ║
║   Five flags. Increasing difficulty. Zero are free.            ║
║                                                                ║
║   Submit flags at the RF Village table.                        ║
║                                                                ║
║   Open Research Institute — openresearch.institute              ║
║   Based on Sharma et al., IEEE A&E Systems Mag, Jan 2026      ║
╚══════════════════════════════════════════════════════════════════╝

SETUP:
    pip install numpy matplotlib
    python lunar_descent_ctf.py --help                # See all options
    python lunar_descent_ctf.py                       # Run the mission
    python lunar_descent_ctf.py --test -p stepwise    # Test a specific profile
    python lunar_descent_ctf.py --test -p all         # Test all three profiles
    python lunar_descent_ctf.py --score               # Check your score and get flags

FLAGS:
    1. RECON (100 pts)          — Explain the bug to village staff
    2. FIRST LIGHT (200 pts)    — Zero tracking gaps on standard profile
    3. SOFT TOUCHDOWN (300 pts) — Land safely on ALL three profiles
    4. SMOOTH OPERATOR (300 pts)— Zero tracking gaps on ALL profiles
    5. MISSION PERFECT (400 pts)— Zero gaps + <0.5% error + land ALL

RULES:
    - Edit ONLY the AutoTracker class (marked with the banner below)
    - Don't change the signal generation, FFT, or altitude estimator
    - Don't hardcode altitudes or cheat the scoring
    - The verify() function checks your fix against hidden test profiles
    - Flags are generated from your solution's actual performance

HINTS (each one costs you 50 points at the village table):
    - Hint 1: Read the state transitions in process_measurement(). 
              What happens to self.state when current_mode changes?
    - Hint 2: The real ISRO system doesn't go back to UNLOCKED during 
              a mode switch. What state would YOU stay in?
    - Hint 3: Think about what you need to verify after switching modes.
              One good measurement in the new mode should be enough to
              confirm the switch worked. 

Good luck. Don't crash. 🌙
"""

import numpy as np
import hashlib
import sys
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple, List

# ═══════════════════════════════════════════════════════════════════
# CONSTANTS — DO NOT MODIFY
# ═══════════════════════════════════════════════════════════════════

C_LIGHT = 299_792_458.0
CHIRP_BW = 240e6
CENTER_FREQ = 35.61e9
SAMPLE_RATE = 10.34e6
NFFT = 8192
FREQ_RES = SAMPLE_RATE / NFFT  # ~1262 Hz/bin

# Processing bandwidth (beat frequency must be in this range)
PROC_BW_LOW = 1.9e6
PROC_BW_HIGH = 4.0e6
GUARD_LOW = 1.7e6
GUARD_HIGH = 4.2e6

PROC_LOW_BIN = int(PROC_BW_LOW / FREQ_RES)
PROC_HIGH_BIN = int(PROC_BW_HIGH / FREQ_RES)
GUARD_LOW_BIN = int(GUARD_LOW / FREQ_RES)
GUARD_HIGH_BIN = int(GUARD_HIGH / FREQ_RES)


# ═══════════════════════════════════════════════════════════════════
# SWEEP MODES — DO NOT MODIFY
# ═══════════════════════════════════════════════════════════════════

@dataclass
class SweepMode:
    mode_id: int
    sweep_time_s: float
    alt_min_m: float
    alt_max_m: float

    @property
    def M(self) -> float:
        """Multiplying constant for altitude: R = M * (up_bin + dn_bin)"""
        import struct
        val = (C_LIGHT * self.sweep_time_s * SAMPLE_RATE) / (4.0 * CHIRP_BW * NFFT)
        return struct.unpack('f', struct.pack('f', val))[0]


def _build_modes(n=12):
    fb_mid = (PROC_BW_LOW + PROC_BW_HIGH) / 2.0
    alts = np.logspace(np.log10(3.0), np.log10(15000.0), n)
    modes = []
    for i, a in enumerate(alts):
        T = 2.0 * CHIRP_BW * a / (C_LIGHT * fb_mid)
        a_lo = PROC_BW_LOW * C_LIGHT * T / (2.0 * CHIRP_BW)
        a_hi = PROC_BW_HIGH * C_LIGHT * T / (2.0 * CHIRP_BW)
        modes.append(SweepMode(i, T, a_lo, a_hi))
    return modes

MODES = _build_modes()


def best_mode_for_altitude(alt_m):
    """Find the mode that puts beat frequency closest to mid-band."""
    fb_mid = (PROC_BW_LOW + PROC_BW_HIGH) / 2.0
    best, best_err = 0, float('inf')
    for m in MODES:
        fb = 2.0 * CHIRP_BW * alt_m / (C_LIGHT * m.sweep_time_s)
        if PROC_BW_LOW <= fb <= PROC_BW_HIGH:
            err = abs(fb - fb_mid)
            if err < best_err:
                best, best_err = m.mode_id, err
    return best


# ═══════════════════════════════════════════════════════════════════
# SIGNAL GENERATION & PROCESSING — DO NOT MODIFY
# ═══════════════════════════════════════════════════════════════════

@dataclass
class PeakResult:
    bin_index: int = 0
    magnitude: float = 0.0
    valid: bool = False


def generate_beat_signal(sweep_time_s, altitude_m, velocity_mps,
                          slope_up=True, snr_db=25.0):
    """Generate beat signal as ADC sees it after analog dechirp."""
    n = int(sweep_time_s * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    fb = 2.0 * CHIRP_BW * altitude_m / (C_LIGHT * sweep_time_s)
    fd = 2.0 * velocity_mps * CENTER_FREQ / C_LIGHT
    f_obs = (fb - fd) if slope_up else (fb + fd)
    sig = np.exp(1j * 2 * np.pi * f_obs * t)
    noise_pwr = 1.0 / (10 ** (snr_db / 10.0))
    noise = np.sqrt(noise_pwr / 2) * (np.random.randn(n) + 1j * np.random.randn(n))
    return sig + noise


def detect_peak(iq_data):
    """FFT + nadir peak detection. Returns PeakResult."""
    n = len(iq_data)
    if n < NFFT:
        padded = np.zeros(NFFT, dtype=complex)
        padded[:n] = iq_data
    else:
        padded = iq_data[:NFFT]
    
    windowed = padded * np.hanning(len(padded))
    spectrum = np.fft.fft(windowed, NFFT)
    mag = np.abs(spectrum[:NFFT // 2])
    
    # Adaptive threshold
    proc_slice = mag[GUARD_LOW_BIN:GUARD_HIGH_BIN]
    if len(proc_slice) == 0 or np.max(proc_slice) == 0:
        return PeakResult(valid=False)
    
    global_max = np.max(proc_slice)
    threshold = global_max * 0.1
    
    # Nadir peak: first valid peak from low frequency
    for i in range(GUARD_LOW_BIN, GUARD_HIGH_BIN):
        if mag[i] > threshold:
            is_peak = True
            if i > 0 and mag[i] < mag[i - 1]:
                is_peak = False
            if i < len(mag) - 1 and mag[i] < mag[i + 1]:
                is_peak = False
            if is_peak:
                return PeakResult(bin_index=i, magnitude=mag[i], valid=True)
    
    return PeakResult(valid=False)


def compute_altitude(up_peak, dn_peak, mode, doppler_limit_bins=50):
    """Altitude from up/down chirp peaks. Returns (alt_m, vel_mps, valid)."""
    if not up_peak.valid or not dn_peak.valid:
        return None, None, False
    if abs(up_peak.bin_index - dn_peak.bin_index) > doppler_limit_bins:
        return None, None, False
    alt = mode.M * (up_peak.bin_index + dn_peak.bin_index)
    fd_hz = (dn_peak.bin_index - up_peak.bin_index) * FREQ_RES / 2.0
    vel = fd_hz * C_LIGHT / (2.0 * CENTER_FREQ)
    return alt, vel, True


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

class TrackState(Enum):
    UNLOCKED = auto()
    LOCKING = auto()
    LOCKED = auto()


class AutoTracker:
    """The autotracking state machine.
    
    This controls which sweep mode (sweep time) is used and manages
    signal acquisition and tracking.
    
    States:
        UNLOCKED — Searching. Trying to find the signal.
        LOCKING  — Candidate found. Verifying with consecutive detections.
        LOCKED   — Tracking. Adjusting sweep time as altitude changes.
    
    The bug is in here somewhere. The system works fine at high altitude
    but loses lock below ~400m during descent. On the real Chandrayaan-3
    mission, the altimeter tracked continuously from 10 km to 3 m.
    
    RULES: You may add new states, new fields, new methods, or modify
    existing methods. You may NOT change the method signature of
    process_measurement() — it must still accept and return the same types.
    """

    def __init__(self):
        self.state = TrackState.UNLOCKED
        self.current_mode = 0
        self.consecutive = 0
        self.required_consecutive = 3
        self.best_mode = 0
        self.best_power = 0.0

    def process_measurement(self, up_peak: PeakResult, dn_peak: PeakResult
                             ) -> Tuple[int, TrackState]:
        """Process one measurement cycle and return (mode_id, state).
        
        Called once per up/down chirp pair. Must decide:
        1. Are we locked on the signal?
        2. Do we need to change sweep modes?
        3. What mode should the next measurement use?
        
        Parameters
        ----------
        up_peak : PeakResult
            Peak detection result from up-chirp
        dn_peak : PeakResult
            Peak detection result from down-chirp
            
        Returns
        -------
        (mode_id, state) : Tuple[int, TrackState]
            The sweep mode to use for the next cycle, and current state
        """

        if self.state == TrackState.UNLOCKED:
            # === SEARCHING ===
            # Try to find the best mode by checking if current mode has signal
            if up_peak.valid:
                if self.best_mode == self.current_mode:
                    self.consecutive += 1
                else:
                    self.best_mode = self.current_mode
                    self.best_power = up_peak.magnitude
                    self.consecutive = 1

                if self.consecutive >= self.required_consecutive:
                    self.state = TrackState.LOCKED
                    self.current_mode = self.best_mode
            else:
                self.consecutive = 0

        elif self.state == TrackState.LOCKED:
            # === TRACKING ===
            if not up_peak.valid:
                # Lost the signal entirely
                self.state = TrackState.UNLOCKED
                self.consecutive = 0
                return self.current_mode, self.state

            # Check if beat frequency is drifting toward guard bands
            # If so, we need a different sweep time
            needs_mode_change = False
            new_mode = self.current_mode

            if up_peak.bin_index > PROC_HIGH_BIN:
                # Beat freq too high — need longer sweep time (higher mode)
                new_mode = min(self.current_mode + 1, len(MODES) - 1)
                needs_mode_change = (new_mode != self.current_mode)
            elif up_peak.bin_index < PROC_LOW_BIN:
                # Beat freq too low — need shorter sweep time (lower mode)
                new_mode = max(self.current_mode - 1, 0)
                needs_mode_change = (new_mode != self.current_mode)

            if needs_mode_change:
                # *** THIS IS WHERE THE PROBLEM LIVES ***
                # When we switch modes, we go back to UNLOCKED and have to
                # re-acquire the signal from scratch. During descent, the
                # altitude is changing fast enough that by the time we
                # re-acquire (3 cycles), we might need ANOTHER mode switch,
                # causing a cascade failure.
                self.state = TrackState.UNLOCKED
                self.current_mode = new_mode
                self.consecutive = 0

        return self.current_mode, self.state

# ═══════════════════════════════════════════════════════════════════
#  END OF EDITABLE SECTION
# ═══════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════
# MISSION SIMULATION — DO NOT MODIFY
# ═══════════════════════════════════════════════════════════════════

@dataclass
class MissionResult:
    """Results from one descent simulation."""
    total_measurements: int = 0
    valid_measurements: int = 0
    max_altitude_tracked: float = 0.0
    min_altitude_tracked: float = float('inf')
    max_error_pct: float = 0.0
    mean_error_pct: float = 0.0
    tracking_gaps: int = 0          # Number of times lock was lost
    longest_gap: int = 0            # Longest gap in measurements
    continuous_from_m: float = 0.0  # Lowest alt with unbroken tracking
    landed_safely: bool = False     # Tracked to below 10m


def run_descent(profile_name="standard", seed=42, verbose=False):
    """Simulate a descent and return mission results."""
    np.random.seed(seed)
    tracker = AutoTracker()
    
    # Build descent profile
    if profile_name == "standard":
        # Chandrayaan-3-like profile
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
    
    elif profile_name == "aggressive":
        # Steeper descent, more mode transitions, reaches touchdown
        times = np.linspace(0, 500, 400)
        profile = []
        for t in times:
            if t < 400:
                alt = 8000.0 * np.exp(-t / 80.0)
                vel = (8000.0 / 80.0) * np.exp(-t / 80.0)
            else:
                # Final approach
                tp = t - 400
                alt = 8000.0 * np.exp(-400 / 80.0) * (1.0 - tp / 100.0)
                vel = 0.5
            alt = max(alt, 3.0)
            vel = max(vel, 0.1)
            profile.append((t, alt, vel))
    
    elif profile_name == "stepwise":
        # Altitude changes rapidly (hover-then-drop pattern)
        # Each drop takes ~10 seconds (not instant) but is still fast
        times = np.linspace(0, 600, 400)
        profile = []
        hover_alts = [5000, 2000, 800, 200, 50, 10]
        segment_duration = 600.0 / len(hover_alts)
        for t in times:
            seg = min(int(t / segment_duration), len(hover_alts) - 1)
            seg_t = t - seg * segment_duration
            
            if seg < len(hover_alts) - 1:
                # First 10% of each segment: rapid descent to next level
                # Remaining 90%: hover at that level
                transition_time = segment_duration * 0.10
                if seg_t < transition_time and seg > 0:
                    # Transitioning from previous altitude
                    prev_alt = hover_alts[seg - 1] if seg > 0 else hover_alts[0]
                    frac = seg_t / transition_time
                    alt = prev_alt + (hover_alts[seg] - prev_alt) * frac
                    vel = abs(hover_alts[seg] - prev_alt) / transition_time
                else:
                    alt = float(hover_alts[seg])
                    vel = 0.5
            else:
                alt = float(hover_alts[-1])
                vel = 0.2
            
            alt = max(alt, 3.0)
            vel = max(vel, 0.1)
            profile.append((t, alt, vel))
    
    else:
        raise ValueError(f"Unknown profile: {profile_name}")
    
    # Run the mission
    results = []
    gap_count = 0
    current_gap = 0
    longest_gap = 0
    was_valid = False
    errors = []
    
    if verbose:
        print(f"\n{'Time':>7s} | {'True Alt':>10s} | {'RAP Alt':>10s} | "
              f"{'Error%':>8s} | {'State':>8s} | {'Mode':>4s}")
        print("-" * 65)
    
    for t, true_alt, true_vel in profile:
        # Use tracker's mode selection, but help it find the right
        # neighborhood when unlocked
        if tracker.state == TrackState.UNLOCKED:
            ideal = best_mode_for_altitude(true_alt)
            tracker.current_mode = ideal
        
        mode = MODES[tracker.current_mode]
        T = mode.sweep_time_s
        
        # Generate signals
        up_beat = generate_beat_signal(T, true_alt, true_vel, True, snr_db=25.0)
        dn_beat = generate_beat_signal(T, true_alt, true_vel, False, snr_db=25.0)
        
        # Detect peaks
        up_peak = detect_peak(up_beat)
        dn_peak = detect_peak(dn_beat)
        
        # Run the autotracker (YOUR CODE)
        mode_id, state = tracker.process_measurement(up_peak, dn_peak)
        
        # Compute altitude
        alt_m, vel_m, valid = compute_altitude(up_peak, dn_peak, mode)
        
        if valid:
            err_pct = abs(alt_m - true_alt) / true_alt * 100 if true_alt > 0 else 0
            errors.append(err_pct)
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
        
        results.append((t, true_alt, alt_m if valid else None, valid, state))
        
        if verbose and (len(results) % 20 == 0):
            alt_str = f"{alt_m:.1f}" if valid else "---"
            err_str = f"{err_pct:.3f}%" if valid else "---"
            print(f"{t:7.1f} | {true_alt:10.1f} | {alt_str:>10s} | "
                  f"{err_str:>8s} | {state.name:>8s} | {mode_id:4d}")
    
    # Handle final gap
    if current_gap > 0:
        gap_count += 1
        longest_gap = max(longest_gap, current_gap)
    
    # Compute mission results
    valid_results = [(t, ta, ra) for t, ta, ra, v, s in results if v]
    
    mr = MissionResult()
    mr.total_measurements = len(results)
    mr.valid_measurements = len(valid_results)
    
    if valid_results:
        mr.max_altitude_tracked = max(ra for _, _, ra in valid_results)
        mr.min_altitude_tracked = min(ra for _, _, ra in valid_results)
        mr.max_error_pct = max(errors) if errors else 0
        mr.mean_error_pct = np.mean(errors) if errors else 0
    
    mr.tracking_gaps = gap_count
    mr.longest_gap = longest_gap
    
    # Error metric: 95th percentile (tolerates rare outliers at extreme altitude)
    if errors:
        mr.max_error_pct = np.percentile(errors, 95)
        mr.mean_error_pct = np.mean(errors)
    
    # Check continuous tracking from bottom
    chain_broken = False
    min_continuous = float('inf')
    for t, ta, ra, v, s in reversed(results):
        if v and not chain_broken:
            min_continuous = ta
        elif not v:
            chain_broken = True
    mr.continuous_from_m = min_continuous if min_continuous < float('inf') else 0
    mr.landed_safely = mr.min_altitude_tracked < 15.0 and mr.continuous_from_m < 100.0
    
    return mr, results


# ═══════════════════════════════════════════════════════════════════
# SCORING AND FLAGS — DO NOT MODIFY
# ═══════════════════════════════════════════════════════════════════

def _flag(seed_str: str) -> str:
    """Generate a flag from a seed string."""
    h = hashlib.sha256(f"ORI-LUNAR-{seed_str}-2026".encode()).hexdigest()[:12]
    return f"FLAG{{lunar_{h}}}"


def score_and_flags():
    """Run all test profiles and compute score + flags.
    
    Flag structure — EVERY flag requires actual code changes:
    
      Flag 1 (100 pts) RECON:           Staff-validated, no hash on screen.
      Flag 2 (200 pts) FIRST LIGHT:     Zero tracking gaps on standard.
      Flag 3 (300 pts) SOFT TOUCHDOWN:   Land safely on ALL profiles.
      Flag 4 (300 pts) SMOOTH OPERATOR:  Zero tracking gaps on ALL profiles.
      Flag 5 (400 pts) MISSION PERFECT:  Zero gaps + <0.5% error + land ALL.
    
    Buggy code scores: 0 / 1300.
    """
    print("\n" + "=" * 65)
    print("  LUNAR DESCENT CTF — SCORING")
    print("=" * 65)
    
    profiles = ["standard", "aggressive", "stepwise"]
    total_score = 0
    flags_earned = []
    all_results = {}
    
    for pname in profiles:
        mr, raw = run_descent(pname, seed=42)
        all_results[pname] = mr
        
        print(f"\n  Profile: {pname}")
        print(f"    Valid measurements: {mr.valid_measurements}/{mr.total_measurements}")
        print(f"    Tracking gaps:      {mr.tracking_gaps}")
        print(f"    Longest gap:        {mr.longest_gap} cycles")
        print(f"    Min alt tracked:    {mr.min_altitude_tracked:.1f} m")
        print(f"    p95 error:          {mr.max_error_pct:.3f}%")
        print(f"    Continuous from:    {mr.continuous_from_m:.1f} m")
        print(f"    Landed safely:      {'YES' if mr.landed_safely else 'NO'}")
    
    mr_std = all_results["standard"]
    mr_agg = all_results["aggressive"]
    mr_step = all_results["stepwise"]
    
    # ── FLAG 1 (100 pts): IDENTIFY THE BUG ──────────────────────
    # Staff-validated ONLY. No hash printed. Participant must walk to
    # the RF Village table and explain the bug to earn this flag.
    # See CTF_README.md for staff validation criteria.
    print(f"\n{'─' * 65}")
    print(f"  FLAG 1 — RECON (100 pts)")
    print(f"  Find the bug in AutoTracker.process_measurement().")
    print(f"  Explain to RF Village staff: which line, what it does")
    print(f"  wrong, and why it causes tracking loss during descent.")
    print(f"  [Flag issued by staff — no hash here]")
    
    # ── FLAG 2 (200 pts): ZERO GAPS ON STANDARD ────────────────
    # Buggy code has 6 gaps on standard. Any fix that eliminates
    # gaps on the gentle profile earns this flag.
    print(f"\n  FLAG 2 — FIRST LIGHT (200 pts)")
    if mr_std.tracking_gaps == 0:
        flag2 = _flag("zero-gaps-standard")
        print(f"  Zero tracking gaps on standard profile!")
        print(f"  {flag2}")
        flags_earned.append(("FIRST LIGHT", 200, flag2))
        total_score += 200
    else:
        print(f"  LOCKED: Need zero tracking gaps on standard profile")
        print(f"  Currently: {mr_std.tracking_gaps} gaps, "
              f"longest: {mr_std.longest_gap} cycles")
    
    # ── FLAG 3 (300 pts): LAND ON ALL PROFILES ─────────────────
    # Buggy code crashes stepwise. Fix must handle abrupt altitude
    # changes, not just smooth descents.
    all_landed = all(all_results[p].landed_safely for p in profiles)
    print(f"\n  FLAG 3 — SOFT TOUCHDOWN (300 pts)")
    if all_landed:
        flag3 = _flag("all-profiles-landed")
        print(f"  Landed safely on ALL profiles!")
        print(f"  {flag3}")
        flags_earned.append(("SOFT TOUCHDOWN", 300, flag3))
        total_score += 300
    else:
        print(f"  LOCKED: Need safe landing on all profiles")
        for p in profiles:
            status = "LANDED" if all_results[p].landed_safely else "CRASHED"
            print(f"    {p}: {status}")
    
    # ── FLAG 4 (300 pts): ZERO GAPS ON ALL PROFILES ────────────
    # This is hard. Stepwise profile forces multi-mode jumps.
    all_zero_gaps = all(all_results[p].tracking_gaps == 0 for p in profiles)
    print(f"\n  FLAG 4 — SMOOTH OPERATOR (300 pts)")
    if all_zero_gaps:
        flag4 = _flag("zero-gaps-all-profiles")
        print(f"  Zero tracking gaps on ALL profiles!")
        print(f"  {flag4}")
        flags_earned.append(("SMOOTH OPERATOR", 300, flag4))
        total_score += 300
    else:
        print(f"  LOCKED: Need zero tracking gaps on ALL profiles")
        for p in profiles:
            mr = all_results[p]
            if mr.tracking_gaps == 0:
                print(f"    {p}: CLEAN")
            else:
                print(f"    {p}: {mr.tracking_gaps} gaps, "
                      f"longest: {mr.longest_gap} cycles")
    
    # ── FLAG 5 (400 pts): MISSION PERFECT ──────────────────────
    # The whole package: zero gaps, accurate, and landed everywhere.
    all_accurate = all(all_results[p].max_error_pct < 0.5 for p in profiles)
    print(f"\n  FLAG 5 — MISSION PERFECT (400 pts)")
    if all_zero_gaps and all_accurate and all_landed:
        flag5 = _flag("mission-perfect-v2")
        print(f"  Zero gaps, <0.5% p95 error, safe landing on ALL profiles.")
        print(f"  You could land on the Moon.")
        print(f"  {flag5}")
        flags_earned.append(("MISSION PERFECT", 400, flag5))
        total_score += 400
    else:
        print(f"  LOCKED: Need zero gaps + <0.5% p95 error + safe landing, ALL profiles")
        for p in profiles:
            mr = all_results[p]
            issues = []
            if mr.tracking_gaps > 0:
                issues.append(f"{mr.tracking_gaps} gaps")
            if mr.max_error_pct >= 0.5:
                issues.append(f"{mr.max_error_pct:.3f}% p95 error")
            if not mr.landed_safely:
                issues.append("crashed")
            status = ", ".join(issues) if issues else "PERFECT"
            print(f"    {p}: {status}")
    
    # ── SUMMARY ────────────────────────────────────────────────
    print(f"\n{'═' * 65}")
    print(f"  TOTAL SCORE: {total_score} / 1300 points")
    print(f"  FLAGS: {len(flags_earned)} / 5")
    print(f"  (Flag 1 is staff-validated at the RF Village table)")
    print(f"{'═' * 65}")
    
    if total_score >= 1200:
        print(f"\n  🌙 You landed on the Moon.")
        print(f"  Show this to the RF Village table for your prize.")
    elif total_score >= 500:
        print(f"\n  Good progress. The spacecraft survived some profiles,")
        print(f"  but mission control wants continuous tracking everywhere.")
    elif total_score > 0:
        print(f"\n  You've started fixing the autotracker. Keep going.")
        print(f"  The stepwise profile is the hard one.")
    else:
        print(f"\n  Score: 0. The autotracker has a bug.")
        print(f"  Start by reading AutoTracker.process_measurement().")
        print(f"  What happens when the altitude changes enough to need")
        print(f"  a different sweep mode?")
    
    return total_score, flags_earned


def run_test(profile_names, verbose=True):
    """Run one or more profiles with verbose output and summary."""
    profiles = profile_names if isinstance(profile_names, list) else [profile_names]
    
    for pname in profiles:
        print(f"\n{'=' * 65}")
        print(f"  LUNAR DESCENT — TEST RUN ({pname} profile)")
        print(f"{'=' * 65}")
        
        mr, results = run_descent(pname, verbose=verbose)
        
        # ASCII visualization of tracking status
        # Each character represents a window of measurements.
        # X if ANY measurement in that window was invalid.
        print(f"\n  Tracking Timeline:")
        print(f"  (. = all valid, X = gap detected, altitude decreasing →)\n")
        
        target_width = 72
        window = max(1, len(results) // target_width)
        line = "  "
        for i in range(0, len(results), window):
            chunk = results[i:i + window]
            all_valid = all(v for (t, ta, ra, v, s) in chunk)
            line += "." if all_valid else "X"
        print(line)
        
        alt_start = results[0][1]
        alt_end = results[-1][1]
        print(f"  {'↑':>2s}{' ' * (len(line) - 5)}{'↑':>2s}")
        print(f"  {alt_start:.0f}m{' ' * (len(line) - 12)}{alt_end:.0f}m")
        
        print(f"\n  Summary:")
        print(f"    Valid measurements: {mr.valid_measurements}/{mr.total_measurements}")
        print(f"    Tracking gaps:      {mr.tracking_gaps} (longest: {mr.longest_gap} cycles)")
        print(f"    p95 error:          {mr.max_error_pct:.3f}%")
        print(f"    Min alt tracked:    {mr.min_altitude_tracked:.1f} m")
        print(f"    MISSION STATUS:     {'LANDED' if mr.landed_safely else 'CRASHED'}")
        
        if not mr.landed_safely:
            print(f"\n  💥 The spacecraft crashed.")
            print(f"  The autotracker lost lock and couldn't recover.")
    
    if any(not run_descent(p, seed=42)[0].landed_safely for p in profiles):
        print(f"\n  Run with --score to see your flags.")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

VALID_PROFILES = ["standard", "aggressive", "stepwise", "all"]


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        prog="lunar_descent_ctf",
        description="""
╔══════════════════════════════════════════════════════════════╗
║  LUNAR DESCENT — BSides San Diego 2026 RF Village CTF      ║
║                                                            ║
║  A radar altimeter guided ISRO's Chandrayaan-3 to the     ║
║  Moon. The autotracker in this model has a bug.            ║
║  Find it. Fix it. Land the spacecraft.                     ║
║                                                            ║
║  5 flags, 1300 points. Zero are free.                      ║
║  Edit ONLY the AutoTracker class.                          ║
║                                                            ║
║  Open Research Institute — openresearch.institute           ║
╚══════════════════════════════════════════════════════════════╝
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  %(prog)s                        Run the default demo (standard profile)
  %(prog)s --test                 Verbose test on standard profile
  %(prog)s --test -p stepwise     Verbose test on stepwise profile
  %(prog)s --test -p all          Verbose test on all three profiles
  %(prog)s --score                Score your fix and earn flags

profiles:
  standard    Smooth Chandrayaan-3-like descent, 10 km → 3 m
  aggressive  Exponential braking, 8 km → 3 m in 500s
  stepwise    Hover-then-drop: 5000/2000/800/200/50/10 m

flags:
  1. RECON (100 pts)           Explain the bug to village staff
  2. FIRST LIGHT (200 pts)     Zero tracking gaps on standard
  3. SOFT TOUCHDOWN (300 pts)  Land safely on ALL profiles
  4. SMOOTH OPERATOR (300 pts) Zero tracking gaps on ALL profiles
  5. MISSION PERFECT (400 pts) Zero gaps + <0.5%% error + land ALL
""",
    )
    
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--test", action="store_true",
        help="Run verbose test (use -p to pick profile, default: standard)",
    )
    mode.add_argument(
        "--score", action="store_true",
        help="Run all profiles and compute score + flags",
    )
    
    parser.add_argument(
        "-p", "--profile",
        choices=VALID_PROFILES,
        default="standard",
        help="Profile to test: standard, aggressive, stepwise, or all (default: standard)",
    )
    
    args = parser.parse_args()
    
    if args.score:
        score_and_flags()
    elif args.test:
        if args.profile == "all":
            run_test(["standard", "aggressive", "stepwise"])
        else:
            run_test(args.profile)
    else:
        # Default: demo run on selected profile (or standard)
        if args.profile == "all":
            run_test(["standard", "aggressive", "stepwise"], verbose=True)
        else:
            run_test(args.profile, verbose=True)


if __name__ == "__main__":
    main()
