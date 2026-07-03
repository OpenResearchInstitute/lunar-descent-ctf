#!/usr/bin/env python3
"""
Reference Solution — Flag 3 (NO GAPS, 400 pts)
================================================

Full fix: per-mode Doppler threshold + velocity rejection.

Earns both Flag 2 AND Flag 3 (zero qualifier rejections,
all landers survive, 900/1000 points).

Two changes from the buggy qualifier:

1. VELOCITY QUALIFICATION (same as Flag 2 fix):
   Reject Doppler-derived velocity when samples_per_sweep < 100.

2. PER-MODE DOPPLER THRESHOLD (the Flag 3 insight):
   Scale the Doppler limit with samples_per_sweep. More samples
   means better peak accuracy, so large real Doppler separations
   are trustworthy at high-sample modes.

Run: python solution_flag3.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lunar_descent_ctf as ctf


def fixed_init(self):
    self.doppler_limit_bins = 40
    self.last_good_altitude = None
    self.last_good_velocity = None


def fixed_qualify(self, rap_output):
    if not rap_output.valid:
        return None, None, False

    mode = ctf.MODES[rap_output.mode_id]
    n_samples = mode.samples_per_sweep

    # FIX #2: Per-mode Doppler limit.
    # More samples → better peak accuracy → larger real Doppler is OK.
    # At mode 0 (14 samples), keep tight limit. At mode 11 (8192),
    # allow large separations for fast descent.
    scaled_limit = max(self.doppler_limit_bins, n_samples // 3)

    if rap_output.up_peak and rap_output.dn_peak:
        doppler_bins = abs(rap_output.up_peak.bin_index -
                          rap_output.dn_peak.bin_index)
        if doppler_bins > scaled_limit:
            return None, None, False

    # FIX #1: Velocity qualification at low modes.
    # Below 100 samples, Doppler-derived velocity is noise.
    velocity = rap_output.velocity_mps
    if n_samples < 100:
        velocity = self.last_good_velocity if self.last_good_velocity is not None else 0.0

    self.last_good_altitude = rap_output.altitude_m
    if n_samples >= 100:
        self.last_good_velocity = rap_output.velocity_mps

    return rap_output.altitude_m, velocity, True


if __name__ == "__main__":
    ctf.MeasurementQualifier.__init__ = fixed_init
    ctf.MeasurementQualifier.qualify = fixed_qualify
    score, flags = ctf.score_and_flags()
    print(f"\nExpected: Flag 2 AND Flag 3 earned (900/1000)")
