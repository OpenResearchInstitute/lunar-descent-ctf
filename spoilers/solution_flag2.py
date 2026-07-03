#!/usr/bin/env python3
"""
Reference Solution — Flag 2 (FIRST LIGHT, 500 pts)
====================================================

Simplistic fix: reject velocity when the mode doesn't have enough
signal samples for reliable Doppler measurement.

Earns Flag 2 (all landers survive) but NOT Flag 3 (qualifier
still rejects valid high-velocity measurements at the top of
aggressive and stepwise descents).

Run: python solution_flag2.py
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

    # Original Doppler check (unchanged)
    if rap_output.up_peak and rap_output.dn_peak:
        doppler_bins = abs(rap_output.up_peak.bin_index -
                          rap_output.dn_peak.bin_index)
        if doppler_bins > self.doppler_limit_bins:
            return None, None, False

    # FIX: Check if this mode has enough samples for velocity
    mode = ctf.MODES[rap_output.mode_id]
    if mode.samples_per_sweep < 100:
        # Not enough samples — altitude is fine, velocity is noise
        self.last_good_altitude = rap_output.altitude_m
        return rap_output.altitude_m, 0.0, True

    self.last_good_altitude = rap_output.altitude_m
    self.last_good_velocity = rap_output.velocity_mps
    return rap_output.altitude_m, rap_output.velocity_mps, True


if __name__ == "__main__":
    ctf.MeasurementQualifier.__init__ = fixed_init
    ctf.MeasurementQualifier.qualify = fixed_qualify
    score, flags = ctf.score_and_flags()
    print(f"\nExpected: Flag 2 earned (500/1000), Flag 3 locked")	
