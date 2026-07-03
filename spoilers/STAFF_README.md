# 🌙 LUNAR DESCENT CTF — STAFF GUIDE
# ⚠️  DO NOT PUT THIS IN THE PUBLIC REPO  ⚠️

BSides San Diego 2026 — RF Village
April 4, 2026 at SDSU

## Setup Checklist
- [ ] Pre-install Python 3 + NumPy on loaner laptops (matplotlib optional)
- [ ] USB sticks with lunar_descent_ctf.py and CTF_README.md
- [ ] QR codes printed: GitHub repo, ORI website, IEEE paper DOI
- [ ] Flag submission sheet printed (flag text → participant name → time)
- [ ] This guide at the village table (paper copy, not on screen)

## Flag Summary

| Flag | Points | How Earned |
|------|--------|------------|
| RECON | 100 | Staff validates verbal explanation |
| FIRST LIGHT | 500 | `--score` prints the flag hash |
| NO GAPS | 400 | `--score` prints the flag hash |

## Validating Flag 1 (RECON)

**No hash is printed on screen.** You issue the flag manually.

Participant must explain the core problem. Accept any version that
covers these four points:

1. **Velocity comes from the difference** of up-chirp and down-chirp
   FFT peak positions (not the sum — altitude uses the sum)
2. **Short sweep times produce few signal samples** — mode 0 has only
   14 real samples in the 8192-point FFT
3. **Few samples = noisy peaks** — peak position error of a few bins
   maps to ±30 m/s of velocity error
4. **The autopilot trusts the velocity** without checking whether the
   mode has enough samples to produce a reliable Doppler measurement

The key insight: **altitude is robust** (sum cancels errors) but
**velocity is fragile** (difference amplifies errors).

### Example good explanations

"The sweep time at low altitude is too short. Mode 0 only gets 14
samples into the FFT. The velocity comes from the difference between
the up and down peaks, and with that few samples the peaks are off
by several bins. Each bin is 2.66 m/s, so you get velocity readings
of 30-60 m/s when the lander is barely moving."

"The Doppler resolution depends on how many real samples you have.
At 3 meters, you have 14. The altitude still works because it uses
the sum of the peaks, but velocity uses the difference, which
amplifies the noise. The qualifier passes it through without checking."

### Example insufficient explanations

"The velocity is wrong at low altitude" — too vague, no mechanism

"The FFT isn't accurate enough" — doesn't explain WHY at low altitude

"There's a bug in the autotracker" — wrong, the RAP works correctly

### Staff flag hash
```
FLAG{lunar_STAFF_ONLY_recon_vel_2026}
```
Write this on the participant's submission sheet when they explain
the bug correctly. Do NOT display this on any screen.

## Hints

Deduct 50 points per hint given. Record which hints were used.

1. **Hint 1 (mild):** The altitude is fine everywhere. What about
   the velocity?

2. **Hint 2 (medium):** Run `--modes` and look at the Samples column.
   What does 14/8192 mean for mode 0?

3. **Hint 3 (strong):** Velocity = difference of two peak positions.
   Each peak has noise of several bins. Each bin = 2.66 m/s.
   What's the velocity error with 14 samples?

### Flag 3 Hints (deduct 50 pts each)

4. **Hint 4 (mild):** After fixing Flag 2, look at the X marks in
   the timeline. The RAP has valid data there — why is the qualifier
   dropping it?

5. **Hint 5 (strong):** The Doppler limit is 40 bins. The aggressive
   profile starts at 167 m/s. How many bins of separation is that?
   Should the qualifier reject it?

## Prize Suggestions

- **1000 pts (NO GAPS):** Custom ORI sticker + bragging rights
  + offered a volunteer position on the FPGA team
- **600 pts (FIRST LIGHT):** ORI sticker + QR code to the GitHub repo
- **100 pts (RECON):** QR code to the GitHub repo + thanks

## Flag 3 (NO GAPS) — What Staff Need to Know

Flag 3 is self-scoring — participants run `--score` and get the hash
if they achieve zero qualifier rejections. No staff validation needed.

**What the participant figured out:** The simplistic Flag 2 fix
(reject velocity at low modes) doesn't touch the fixed 40-bin Doppler
threshold. That threshold rejects valid RAP data during fast descent
because the real Doppler separation exceeds 40 bins:
- Aggressive profile: 167 m/s at the top → 63 bins separation
- Stepwise drops: 138-393 m/s → 52-148 bins separation

**The fix:** Scale the Doppler limit per mode. More signal samples
= better peak accuracy = larger real Doppler separations are
trustworthy. Common approaches: `max(40, n_samples // 3)` or
scale proportionally with sweep time.

**One fixed threshold, two failure modes:**
- Too loose at the bottom (passes noise velocity) → Flag 2
- Too tight at the top (rejects real velocity) → Flag 3

## Timing

The CTF runs all day alongside workshop modules and talks. It's
self-paced. Staff attention needed only for:
- Flag 1 (RECON) validation — ~2 min per participant
- Prize distribution
- Answering setup questions (pip install, etc.)

Most participants with signal processing background solve Flag 2 in
20-30 minutes. Flag 3 requires noticing the X marks in the timeline
after fixing Flag 2, and understanding the Doppler threshold — add
another 15-20 minutes. Complete beginners may need all hints and
60+ minutes for Flag 2.

## Common Questions from Participants

**"Is the autotracker broken?"**
No. The RAP (signal generation, FFT, peak detection, autotracking,
altitude computation) all work correctly. The bug is in the
MeasurementQualifier — what it passes to the autopilot.

**"Why does the autopilot crash at high altitude too?"**
It doesn't. Look at the Mode column — CRASH only happens when
altitude < 30m AND velocity > 20 m/s. At high altitude, large
velocity is expected (the lander IS descending fast).

**"Can I change the autopilot?"**
No. The fix goes in MeasurementQualifier.qualify() only.

**"What's the 'right' fix?"**
Any fix that lands all three profiles works for Flag 2. Common approaches:
- Reject velocity when mode < 3 (simple, effective)
- Clamp velocity to 0 when samples_per_sweep < 100
- Use last_good_velocity when current mode is unreliable
- Scale a confidence threshold by samples_per_sweep

For Flag 3, participants also need to scale the Doppler threshold
so valid high-velocity measurements aren't rejected.

## Technical Reference

Velocity resolution per bin: 2.66 m/s
(FREQ_RES / 2) × c / (2 × fc) = 1262.7/2 × 299792458 / (2 × 35.61e9)

Empirical velocity noise by mode:
  Mode 0 (14 samples):  ±30 m/s — garbage
  Mode 1 (30 samples):  ±10 m/s — dangerous
  Mode 2 (63 samples):  ±3 m/s  — marginal
  Mode 3 (130 samples): ±1 m/s  — good
  Mode 4+ (267+ samples): < 1 m/s — reliable

Autopilot crash condition: altitude < 30m AND |velocity| > 20 m/s
