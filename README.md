# LUNAR DESCENT — BSides San Diego 2026 RF Village CTF

A capture-the-flag (CTF) challenge based on a real bug in a radar altimeter state machine.

The radar altimeter processor that guided ISRO's Chandrayaan-3 to a soft lunar landing runs on a single FPGA. This CTF uses a Python reference model of that system with a **deliberately introduced bug** in the autotracking state machine. Participants must find the bug, understand why it causes the spacecraft to crash, and fix it.

## For Participants

```bash
pip install numpy matplotlib
python lunar_descent_ctf.py          # Run the mission — watch it fail
python lunar_descent_ctf.py --test   # Quick test with verbose output
python lunar_descent_ctf.py --score  # Check your score and earn flags
```

**Rules:**
- Edit ONLY the `AutoTracker` class (clearly marked in the source)
- Don't change signal generation, FFT, peak detection, or altitude estimator
- Don't hardcode altitudes
- Submit flags at the RF Village table

## Five Flags

| Flag | Points | Challenge |
|------|--------|-----------|
| RECON | 100 | Find the bug. Explain it to village staff. |
| FIRST LIGHT | 200 | >80% valid measurements on standard profile |
| SMOOTH OPERATOR | 200 | Zero tracking gaps on standard profile |
| SOFT TOUCHDOWN | 300 | Land safely on ALL three test profiles |
| MISSION PERFECT | 500 | Zero gaps + <0.5% error + landing on ALL profiles |

**Total: 1300 points**

## Difficulty Curve

- **300 pts** — Reading the code and understanding the problem. Most anyone with Python experience.
- **600 pts** — A reasonable fix. Requires understanding state machines. ~30 min.
- **800 pts** — Smart mode prediction. Requires understanding the physics. ~45 min.
- **1300 pts** — Genuinely hard. Requires sophisticated state machine design with predictive mode switching and graceful degradation. Could take 60+ min.

## For Village Staff

### Setup
- Pre-install Python + NumPy + matplotlib on loaner laptops
- Have the file available via QR code (GitHub) and USB stick
- Print the flag submission sheet (flag text and participant name)

### Validating Flag 1 (RECON)
Participant must explain: *"The autotracker drops back to UNLOCKED when it needs to switch sweep modes. The 3-consecutive-detection re-acquisition requirement means it takes several cycles to re-lock, during which the altitude keeps changing. If it changes enough to need ANOTHER mode switch, you get a cascade failure and the tracker never re-acquires."*

Accept any version of this that shows they understand the cascade/re-acquisition problem. The specific line is `self.state = TrackState.UNLOCKED` inside the `needs_mode_change` block.

### Hints (deduct 50 pts each if given)
1. Look at what happens to `self.state` when `needs_mode_change` is True.
2. The real ISRO system doesn't go back to UNLOCKED during a mode switch.
3. What if you stayed LOCKED and just changed the sweep time? What would you need to verify?

### Prize Suggestions
- **1300 pts (MISSION PERFECT):** Custom ORI sticker + bragging rights + offered a volunteer position on the FPGA team
- **800 pts (SMOOTH OPERATOR + SOFT TOUCHDOWN):** ORI sticker
- **300 pts (RECON + FIRST LIGHT):** QR code to the GitHub repo

### Timing
The CTF can run all day alongside the workshop modules. It's self-paced and doesn't require staff attention except for Flag 1 validation and prize distribution.

## Test Profiles

| Profile | Character | What It Tests |
|---------|-----------|---------------|
| standard | Chandrayaan-3-like smooth descent, 10 km → 3 m | Basic mode transitions during gradual altitude change |
| aggressive | Exponential braking, 8 km → 3 m in 500s | Rapid mode transitions with high descent rate |
| stepwise | Hover-then-drop, 5000/2000/800/200/50/10 m | Abrupt altitude changes requiring fast mode adaptation |

## Connection to Real Engineering

The bug is pedagogically motivated but the *pattern* is real. State machines that drop back to initialization during mode transitions are a common source of intermittent tracking failures in:
- Communication protocol state machines (TLS, 802.11 roaming)
- Radar and sonar tracking systems
- Motor control and servo loops
- Protocol state machines in network equipment

The fix is maintaining state through transitions while verifying the new mode works — is a fundamental design pattern.

## Source

Based on: Sharma et al., "FPGA Implementation of a Hardware-Optimized Autonomous Real-Time Radar Altimeter Processor for Interplanetary Landing Missions," IEEE A&E Systems Magazine, Vol. 41, No. 1, January 2026.
