# 🌙 LUNAR DESCENT — BSides San Diego 2026 RF Village CTF

A capture-the-flag challenge based on a real signal processing problem in a radar altimeter.

ISRO's KaRA radar altimeter guided Chandrayaan-3 to a soft lunar landing on 23 August 2023. The Radar Altimeter Processor (RAP) computes altitude and velocity from FMCW chirp signals, running on a single Xilinx Virtex-5 FPGA. This CTF uses a Python model of that system — faithful to the published paper — where the altimeter feeds a landing autopilot. The altimeter works perfectly. The autopilot keeps crashing. Why?

## For Participants

```bash
pip install numpy matplotlib
python lunar_descent_ctf.py --help               # See all options
python lunar_descent_ctf.py                      # Run the mission — watch it crash
python lunar_descent_ctf.py --modes              # See the sweep mode table
python lunar_descent_ctf.py --test -p all        # Test all three profiles
python lunar_descent_ctf.py --score              # Score your fix and earn flags
```

**Rules:**
- Edit ONLY the `MeasurementQualifier` class (clearly marked in the source)
- Don't change the RAP, signal generation, autopilot, or scoring
- The qualifier decides what the autopilot sees — fix it there
- Submit flags at the RF Village table

## Two Flags

**Zero flags are free.** The buggy code scores 0 / 600 out of the box.

| Flag | Points | Challenge |
|------|--------|-----------|
| 🔍 RECON | 100 | Explain the bug to village staff (no hash on screen) |
| 🔭 FIRST LIGHT | 500 | Land all three profiles without crashing |

**Total: 600 points**

## The Scenario

The KaRA radar altimeter was tested on helicopters and aircraft at altitudes above 50 meters. It worked flawlessly. Field test performance met all mission specifications.

The altimeter is now integrated with a landing autopilot that uses both altitude and velocity measurements for thrust control during final approach. In simulation, the autopilot crashes the lander every time below 15 meters altitude. The altitude readings are fine — sub-meter accuracy all the way to touchdown. Something else is killing the lander.

You need to find out what's going wrong and fix the measurement qualification logic so the autopilot can land safely.

## Difficulty Curve

- **0 pts** — Running the code unmodified. The default run shows OK status all the way down until the final approach, then CRASH. 
- **100 pts** — Explaining the problem to staff. 
- **600 pts** — Fixing the `MeasurementQualifier` so it can land. 

### Validating Flag 1 (RECON)
**No hash is printed on screen for Flag 1.** Staff issue the flag manually.

### Timing
The CTF can run all day alongside the workshop modules and talks. It's self-paced and doesn't require staff attention except for Flag 1 validation and prize distribution. Most participants who understand FMCW radar or FFTs could solve it in less than an hour.

## Test Profiles

| Profile | Character | What It Tests |
|---------|-----------|---------------|
| standard | Chandrayaan-3-like smooth descent, 10 km → 3 m, with altitude excursion at 20 m (thruster anomaly or drifting over crater) | Landing |
| aggressive | Fast exponential braking, 10 km → 3 m in 400 s | Rapid mode transitions at high altitude + Landing |
| stepwise | Hover at guard band boundaries (9851/4795/2334/553/131/31/5 m), drop between them | Mode transitions + Low Hover |

## The Physics

The RAP uses FMCW radar. Up-chirp and down-chirp signals produce beat frequencies:

- **f_up = fb − fd** (up-chirp)
- **f_dn = fb + fd** (down-chirp)

**Altitude** comes from the **sum**: R = M × (f_up_index + f_dn_index). This is robust because peak position errors in the two chirps partially cancel when summed.

**Velocity** comes from the **difference**: fd = (f_dn_index − f_up_index) × freq_res / 2. This is fragile because peak position errors **add** when differenced.

The FFT is always 8192 points, but the number of real signal samples depends on the sweep time:
- Mode 12 (high altitude): 8192 samples → full FFT
- Mode 0 (3 m altitude): 14 samples → 99.8% zero-padding

## Connection to Real Engineering

The problem is pedagogically framed but the pattern is real. Sensor qualification, knowing *when to trust* a measurement and when to reject it, is important.

- Radar and sonar tracking systems (Doppler reliability vs integration time)
- GPS/INS integration (knowing when satellite geometry is too poor to trust)
- Medical imaging (SNR-dependent confidence in measurements)
- Autonomous vehicle sensor fusion (camera vs lidar vs radar confidence)

The paper mentions "three sample qualification logics to generate the final altitude" without detailing them. What are those logics? Will some of those qualification logics help solve this CTF?

## Source

Based on: Sharma et al., "FPGA Implementation of a Hardware-Optimized Autonomous Real-Time Radar Altimeter Processor for Interplanetary Landing Missions," IEEE A&E Systems Magazine, Vol. 41, No. 1, January 2026. DOI: 10.1109/MAES.2025.3595090

Open Research Institute https://openresearch.institute/getting-started
