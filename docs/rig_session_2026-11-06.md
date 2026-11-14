# Rig recording session, 6 Nov 2026: handoff for Claude Code

Written by Claude (claude.ai), which guided the lab session step by step from photos of the lab screen. The user ran every step on the hardware. Each statement is tagged:
**Verified** (seen in screenshots or measured), **Inference** (strongly supported, not directly measured), **Hypothesis** (plausible, untested), or **Unknown** (ask the user).

---

## 0. Status

- All **16 planned recordings** were made on the real rig. Every file is at 1 kHz, with no dropped samples (§7).
- The user has copied the CSVs into the repository. The expected location is `data/rig/`; confirm it.
- **Nothing has been evaluated yet.** `data/rig/metadata.txt` has not been written.
- Several things differ from what the protocol (README §12, `scripts/rig/`) assumed. Read §2–§5 before touching the evaluation:
  1. The lab PC runs **Windows XP + MATLAB R2009a**, so the scripts had to be rewritten. The as-run versions are in `as_run_scripts/`.
  2. The model was built from the vendor **Cart Control** model. The centring loop is a plain **Transfer Fcn**, because the vendor PID block refuses small gains.
  3. The logged data are already in **metres and radians**, not encoder counts.
  4. The logged angle reads **π when the pendulum hangs**, not 0.
  5. The two-tone amplitude was **0.25 V for run4 trial 1** and **0.35 V for run4 trials 2–3 and run7**. The scripts still say 0.25.
- Your job is in §11: verify the data, write the metadata, run the evaluation, and bring the repo's rig protocol and scripts in line with what was actually done.

### Quick facts

| Item | Value |
|---|---|
| Columns | `time_s, cart_pos, pend_angle, command_v` |
| Sample time | 0.001 s (fixed-step `ode1`); N = run length × 1000 rows |
| `cart_pos` | m; 0 = start position (cart centred by hand); **+x = towards the end of the track with the "Feedback" logo** |
| `pend_angle` | rad; **= π at model start with the pendulum hanging still**; deviation from hanging is `phi = pend_angle − π` |
| `command_v` | V; **total** command into the DAC block = excitation + centring loop; positive drives the cart towards +x |
| Centring loop | `G(s) = (2.03 s + 1.5)/(0.02 s + 1)` applied to `(0 − x)`, i.e. Kp = 1.5 V/m, Kd = 2.0 V·s/m, τ = 0.02 s |
| Resolution (Inference) | cart 0.078 m / 1024 = 7.6171875e-5 m per count; angle 2π/2048 = 3.0680e-3 rad per count |

---

## 1. Ground rules

1. **Treat the CSVs as raw data.** Never edit them in place. Derived or cleaned data goes elsewhere.
2. Where this document says **Unknown**, ask the user (§10) instead of guessing.
3. The pendrive came from a PC infected with a USB "shortcut" virus (§9.8). Before any commit, make sure no `.lnk`, `.vbs`, `.exe`, `.inf`, `autorun*`, `desktop.ini`, `Thumbs.db`, `._*` or hidden folders from the pendrive ended up in the repo.
4. Commit only when the user asks.

---

## 2. Hardware and software actually used

| Item | Protocol assumed | Actual (Verified unless marked) |
|---|---|---|
| Rig | "Feedback 33-936S" | Labels read **Digital Pendulum Mechanical Unit 33-200** and **Digital Pendulum Controller 33-201**. The controller has a Power switch, a green **Start** (enables the drive), a red **Stop** and a Ready LED. |
| Lab PC OS | Windows 7 (as the user first said) | **Windows XP** (`C:\Documents and Settings\...`, XP dialogs) |
| MATLAB | not specified | **7.8.0 (R2009a)**: Real-Time Windows Target, Open Watcom compiler (`wcc386`/`wlink`), Simulink **External** mode |
| I/O | not specified | Advantech **PCI-1711**, through Feedback blocks "PCI1711 Lab I/O Board – Feedback Encoder" and "… – Feedback DAC" |
| Vendor software | not specified | Feedback "PendulumModels", **version 1.10**. Real-time models: Cart Control, Cart Ident, CraneIdent, CraneStab, InvPendIdent, PendStabPD, PendSwingUp, PendulumFriction, SwingHoldPendulum, SwingHoldPendulumExtra, UpDownPendulum, PendulumTest |
| MATLAB current folder | not specified | `C:\Feedback\Pendulum\Examples`. The CSVs, `obsrec.rwd` and `obsrec_rtwin\` were written here. |
| Model file | not specified | `obsrec.mdl`, saved in the vendor folder named "Real-time models" (next to `CartControl.mdl`) |
| Lab PC clock | not specified | Set to **10 Oct 2012**, so all file timestamps from that PC are meaningless |
| Transfer | pendrive | pendrive (the lab PC is infected, §9.8) |

The originals are still on the lab PC in `C:\Feedback\Pendulum\Examples\*.csv`. The first five recordings are also in `C:\Documents and Settings\NIT\Desktop\rig_pendrive\`.

---

## 3. The model as run: `obsrec.mdl` (Verified)

It was built from the vendor **Cart Control** real-time model. That model was opened fresh, without the lab assistant's unsaved demo edits, and saved as `obsrec`.

```
x [m] (encoder) ──┬──────────────────────────────────────────────► Mux1 in1
                  └──(−)┐
0 (set point) ───────(+)┴ Sum ─► e = −x ─► G(s) ─┐
                                                  ├─ Add ─► u_total ─► Feedback DAC
u_exc(t) (From Workspace) ────────────────────────┘          ├────────► Mux1 in3
                                                              └────────► vendor scope "control signal"
θ [rad] (encoder) ─► vendor Terminator; branch ─────────────────────────► Mux1 in2
Mux1 ─► To Workspace "simout" (Structure With Time)
```

| Block | Setting |
|---|---|
| Set-point subsystem (vendor: 3 Sine Waves → Add → **Slider Gain** → Saturation) | Slider Gain changed 0.1 → **0** (Low 0, High 0.2), so the set point is always 0 |
| Sum (vendor) | top input − position, bottom input + set point, so e = −x |
| Vendor "PID Controller with D part filter" | **Deleted.** Its mask (P+I/s+Ds; vendor values P 27.8, I 50, D 3.9) forces P ∈ [10, 50]: "Proportional Gain is out of the range [10..50] and is set to minimum now". |
| **Transfer Fcn** (Simulink/Continuous) | Numerator `[2.03 1.5]`, Denominator `[0.02 1]`, so G(s) = Kp + Kd·s/(τs+1) with Kp 1.5, Kd 2.0, τ 0.02 |
| Add | `++`: in1 = Transfer Fcn output, in2 = From Workspace |
| From Workspace | Data `u_runX` (§5); Sample time 0; Interpolate data on; **Form output after final data value by: Setting to zero** |
| Feedback DAC (vendor) | Driven by the Add output. **Never opened**: its scaling, saturation and initial/final values are Unknown. |
| Mux1 | 3 inputs: 1 position, 2 angle, 3 Add output. The command was first wired to input 1 by mistake, then rewired; the final order is confirmed by the data (§4). |
| To Workspace | Variable `simout`, Limit data points `inf`, Decimation 1, Sample time −1, **Structure With Time** |
| Solver (Ctrl+E) | Fixed-step, **ode1**, step **0.001** s, start 0. Periodic sample-time constraint Unconstrained, single-tasking. Stop time set per run in the toolbar. |
| External Mode Control Panel → Signal & Triggering | Blocks selected: To Workspace and the vendor scope. Source **manual**, mode **one-shot**, **Duration = run length × 1000 samples** (default was 100000), Delay 0, **Arm when connecting to target** ticked. |

Notes:
- Saving shows "Block diagram 'obsrec' contains disabled library links". This is inherited from the vendor model and harmless.
- From Workspace data is **compiled into the real-time module**, so every change of input variable or amplitude needs a rebuild (Ctrl+B). The user rebuilt (Ctrl+D, Ctrl+S, Ctrl+B) every time.

---

## 4. Signal conventions and units

Export call used for every file: `export_run(name, simout.time, simout.signals.values, [1 2 3])`.

| Column | Unit and convention | Status and evidence |
|---|---|---|
| `time_s` | Model time, Δt = 1.000 ms, N = T×1000 rows | **Verified** Δt: every printout shows median = largest step = 1.000 ms. Whether `t[0]` is 0 is **Unknown**; check it. |
| `cart_pos` | m, already scaled by the vendor block. 0 = position at model start (cart centred by hand, by eye). **+x = towards the "Feedback" logo end** (the left end seen from the front of the rig). | **Verified** direction: with the motor off, a hand push towards the logo end gave +0.046 m then +0.214 m. **Inference** resolution 0.078/1024 m: printed values are integer multiples (e.g. 0.116543 = 1530 counts, 0.189896 = 2493 counts). |
| `pend_angle` | rad. **Exactly π (3.14159) at model start with the pendulum hanging still**, so the vendor convention is probably 0 = upright (**Inference**). Use `phi = pend_angle − π`. | **Verified** π at rest: run1's range is [3.14159, 3.14159]. **Inference** resolution 2π/2048: deviations are integer multiples (e.g. run3 max = 91 counts, run2 trial3 max = 169 counts). **Sign relative to x: Unknown**; see Task C. |
| `command_v` | V, the **total** value into the Feedback DAC block: `u_exc + G(s)·(0 − x)`. A double, not quantised. **Positive drives the cart towards +x.** | **Verified** by two independent observations: (a) with this sign the cart held still at zero input and stayed bounded and repeatable under every excitation (16 recordings, none reaching a stop); (b) the opposite sign (numerator `[-2.03 -1.5]`) accelerated the cart into an end stop. **Consistency check:** run1's command maximum, 0.007731 V, equals 101.5 × 7.6172e-5. That is G's high-frequency gain (2.03/0.02) times one count, when x stepped to −1 count. |

**Not measured:** the voltage at the amplifier, newtons per volt (2.0 N/V is the manual value quoted in the protocol), cart and pendulum masses, and lengths.

---

## 5. Inputs actually applied

The definitions are in `as_run_scripts/rig_inputs.m`. These are the amplitudes actually used:

| Variable | Shape | Amplitude used | Used for | Peak excitation |
|---|---|---|---|---|
| `u_run1` | 0 V, 20 s | n/a | run1 | 0 |
| `u_run2` | 0 V, 60 s | n/a | run2 ×3 | 0 (pendulum lifted by hand) |
| `u_run3` | Staircase: every 3 s, +a for 0.5 s, rest 1 s, −a for 0.5 s, rest 1 s; a = 0.10 + 0.05·floor(t/3) for 15 levels (0.10…0.80 V); 45 s | n/a | run3 | 0.80 V |
| `u_run4` | A·(2 sin 1.5t + 1.2 cos 3t), 50 s; asymmetric: max +1.617A, min −3.2A | **0.25** (trial 1), **0.35** (trials 2, 3) | run4 | 0.80 V / 1.12 V |
| `u_run5` | A·3·e^(−t/8)·sin(1.6t), 60 s | 0.40 | run5 ×3 | 1.06 V at t ≈ 1 s |
| `u_run6_square`, `_multisine`, `_chirp`, `_pulse` | Held-out H1–H4, 12 s each | 0.70 | run6 | 0.70 V |
| `u_run7` | = `u_run4` at **0.35** | 0.35 | run7 | 1.12 V |

The 0.35 V arrays were created at the MATLAB prompt; `rig_inputs.m` still says 0.25:
```matlab
A_twotone = 0.35; t = (0:0.001:50)'; u_run4 = [t, A_twotone*(2*sin(1.5*t) + 1.2*cos(3*t))];
u_run7 = u_run4;
```
That line printed a peak of 1.1200.

**Evidence that run7 used 0.35 (Verified):** run7's command minimum (−0.362) and cart range (−0.244/+0.135) match run4 trials 2–3 (−0.360/−0.347; about −0.25/+0.13), not trial 1 (−0.225; −0.190/+0.067).

**Why 0.35 rather than the protocol's 0.40:** the rule was to raise to 0.40 if trial 1 stayed within ±0.25 m. It did, reaching −0.19 m, but the real cart travelled about 1.5× more than the simulation predicted (§8.3), so 0.40 was expected to reach about ±0.35 m. With 0.35 V the cart reached −0.25/+0.13 m.

No lead-in was added. A 3 s zero lead-in was proposed and the user declined it.

---

## 6. Procedure as run, per recording (Verified)

1. Set the From Workspace **Data**. Set **Duration = T×1000** and click Apply. Set the **stop time = T + 5 s**; the first test used T + 1.
2. Ctrl+D, Ctrl+S, Ctrl+B, then wait for `### Successful completion of Real-Time Workshop build procedure for model: obsrec`.
3. Centre the cart by hand and let the pendulum hang completely still.
4. **Connect To Target → green Start → Run**. Run1 alone used Connect → Run → green Start, so its drive was enabled 1–2 s after model start; its input was zero.
5. Once T passed the input length, press **red Stop** on the controller, **before** the model's stop time (§9.4). This was instructed for every recording, and no further run-aways were reported.
6. Run `export_run(...)` at the prompt.

| File(s) | Data | T (s) | Duration | Stop time |
|---|---|---|---|---|
| run1_rest_trial1 | `u_run1` | 20 | 20000 | 25 |
| run3_deadzone_trial1 | `u_run3` | 45 | 45000 | 50 |
| run2_freeswing_trial1–3 | `u_run2` | 60 | 60000 | 65 |
| run4_twotone_trial1–3 | `u_run4` | 50 | 50000 | 55 |
| run5_fading_trial1–3 | `u_run5` | 60 | 60000 | 65 |
| run6_heldout_square / multisine / chirp / pulse | `u_run6_*` | 12 | 12000 | 17 |
| run7_mass_step_twotone | `u_run7` | 50 | 50000 | 55 |

**Run 2 (free swing):** right after Run, the user took the bob, lifted it to about 15°, 24° and 29° in the three trials, and released it. The first seconds of each file contain hand contact.

**Run 7:** the extra mass was taped to the pendulum rod **before** the run (instructed: about halfway down, rigid, no dangling). What was attached and where are **Unknown**: a key-ring hook, possibly with keys. Ask the user (§10).

---

## 7. The 16 recordings (Verified: printed by `export_run` on the lab PC)

Every file has the header `time_s,cart_pos,pend_angle,command_v`, and median step = largest step = 1.000 ms. Values are as printed (`%g`, 6 significant digits). The command minimum may be truncated where the screen line was cut off. Rows are in recording order.

| # | File | Rows | Span (s) | cart_pos min / max (m) | pend_angle min / max (rad) | command_v min (V) | Notes |
|---|---|---|---|---|---|---|---|
| 1 | run1_rest_trial1.csv | 20000 | 19.999 | −7.61719e-5 / −0 | 3.14159 / 3.14159 | 0 (max 0.007731) | Zero input, hands off |
| 2 | run3_deadzone_trial1.csv | 45000 | 44.999 | −0.116543 / 0.0383145 | 2.84707 / 3.42078 | −0.80126 | |
| 3 | run2_freeswing_trial1.csv | 60000 | 59.999 | −0.000152344 / −0 | 2.87775 / 3.40851 | −0.00763996 | Lifted ≈15° |
| 4 | run2_freeswing_trial2.csv | 60000 | 59.999 | −0.00220898 / 7.61719e-5 | 2.72742 / 3.56497 | (cut off) | ≈24° |
| 5 | run2_freeswing_trial3.csv | 60000 | 59.999 | −0.00304688 / 7.61719e-5 | 2.62924 / 3.66008 | (cut off) | ≈29° |
| 6 | run4_twotone_trial1.csv | 50000 | 49.999 | −0.189896 / 0.0674883 | 2.98206 / 3.26738 | −0.22483 | A = 0.25 |
| 7 | run4_twotone_trial2.csv | 50000 | 49.999 | −0.249691 / 0.127588 | 2.9115 / 3.31033 | −0.360072 | A = 0.35 |
| 8 | run4_twotone_trial3.csv | 50000 | 49.999 | −0.25091 / 0.132768 | 2.9207 / 3.30726 | −0.346676 | A = 0.35 |
| 9 | run5_fading_trial1.csv | 60000 | 59.999 | −0.174129 / 0.293033 | 3.04035 / 3.2459 | −0.245193 | |
| 10 | run5_fading_trial2.csv | 60000 | 59.999 | −0.173824 / 0.291205 | 3.04035 / 3.24897 | −0.250809 | |
| 11 | run5_fading_trial3.csv | 60000 | 59.999 | −0.173063 / 0.294328 | 3.03421 / 3.24284 | −0.250289 | |
| 12 | run6_heldout_square.csv | 12000 | 11.999 | −0.250682 / 0.244207 | 2.96979 / 3.28579 | −0.33797 | |
| 13 | run6_heldout_multisine.csv | 12000 | 11.999 | −0.153105 / 0.089959 | 2.59856 / 3.67542 | −0.388917 | Largest swing (≈ ±31°) |
| 14 | run6_heldout_chirp.csv | 12000 | 11.999 | −0.151201 / 0.216557 | 2.96979 / 3.29806 | −0.299521 | |
| 15 | run6_heldout_pulse.csv | 12000 | 11.999 | −0.0702305 / 0.0489785 | 3.00047 / 3.28579 | −0.7142 | One digit of the cart minimum was partly hidden by the cursor |
| 16 | run7_mass_step_twotone.csv | 50000 | 49.999 | −0.244436 / 0.134748 | 2.9207 / 3.30113 | −0.362458 | A = 0.35, extra mass attached |

Not exported:
- the test runs during set-up (their `simout` was overwritten);
- one free-swing attempt in which the pendulum was never lifted, which was repeated.

---

## 8. Observations so far (from printouts and photographed plots; verify)

1. **The loop shrinks the command a lot.** The derivative term (2 V·s/m) opposes motion, so total-command peaks are much smaller than the excitation peaks:

   | Input | Excitation peak | Total command peak |
   |---|---|---|
   | Two-tone, A = 0.25 | 0.80 V | −0.22 V |
   | Two-tone, A = 0.35 | 1.12 V | −0.35/−0.36 V |
   | Held-out square, chirp, multisine | 0.70 V | −0.30…−0.39 V |
   | Held-out pulses (too short for the loop to react) | 0.70 V | −0.71 V |

   Much of the identification data therefore sits at **0.2–0.4 V, close to the break-away level**, so stick–slip friction will matter. The data is also **closed-loop**: `command_v` partly depends on x. Encoder noise is only quantisation, so the bias should be small. If a method is sensitive to it, the excitation u_exc (Task B) can serve as an instrument.

2. **Break-away (Inference, from a photo of the run3 plot):** almost no motion at the 0.10 and 0.15 V levels; clear steps from about **0.20–0.25 V** (t ≈ 6–9 s); more travel in −x. Compute it properly (Task D).

3. **The real cart travels more than the repo's simulation predicted.** The protocol's simulation used dead-zone +0.15/−0.12 V, Coulomb friction 0.35 N, 2 N/V and the same loop.

   | Input | Simulated travel, with loop (m) | Measured (m) |
   |---|---|---|
   | run3 staircase to 0.8 V | −0.05 … +0.04 | −0.117 … +0.038 |
   | run4 two-tone A = 0.25 | −0.13 … 0.00 | −0.190 … +0.067 |
   | run4 two-tone A = 0.40 (sim) / 0.35 (measured) | −0.23 … +0.10 | −0.251 … +0.133 |
   | run5 fading 0.40 | −0.12 … +0.24 | −0.174 … +0.294 |
   | Held-out square 0.7 V | −0.13 … +0.16 | −0.251 … +0.244 |
   | Held-out chirp 0.7 V | −0.07 … +0.14 | −0.151 … +0.217 |
   | Held-out multisine 0.7 V | ±0.05 | −0.153 … +0.090 |
   | Held-out pulses 0.7 V | ±0.03 | −0.070 … +0.049 |

   So the simulated actuator and plant are too weak: N/V too low and/or friction too high. Calibrate (Task D).

4. **Repeatability is high.** Trials of the same run agree within a few millimetres and a few angle counts.

5. **Free swing:** the cart moved ≤ 3 mm (friction plus the loop held it), so run2 is effectively a fixed-pivot free decay at about ±15°, ±24° and ±29°.

6. **Multisine** gave the largest pendulum swing of any run, about ±31°. Probably one of its tones (0.7, 2.2, 3.7, 5.3 rad/s) is near ω_n; check against ω_n from run2.

7. **The mass step looks small.** Compare run7 with run4 trials 2–3 (same input):
   - cart: −0.244/+0.135 m against −0.250/+0.130 m;
   - angle maximum: 3.30113 against 3.31033/3.30726, i.e. 2–3 counts;
   - angle minimum: identical to trial 3.

   Quantify it with an ω_n shift (spectrum or model fit), not with ranges. If it's undetectable, tell the user that a repeat with a heavier, weighed mass would be needed.

---

## 9. Incidents during the session (chronological)

### 9.1 Platform
The lab PC is XP + R2009a, not Windows 7 with a modern MATLAB. The original scripts used implicit expansion (the multisine line) and `table`/`writetable`, which R2009a lacks. The rewritten versions (`bsxfun`; `fopen`/`fprintf`) were used for **all** recordings. They give identical inputs and an identical CSV layout and are in `as_run_scripts/`.

### 9.2 Vendor PID mask
It refuses P < 10, so the PID block was replaced by the Transfer Fcn (§3).

### 9.3 Checksum mismatch
The first Connect after the first build failed with "Checksum mismatch. Target code needs to be rebuilt." Fix: Ctrl+D, Ctrl+S, Ctrl+B, then Connect without touching the diagram. It didn't recur.

### 9.4 Run-aways and a sign error

1. **First powered test** (zero input, numerator `[2.03 1.5]`, stop time 21, Duration 20000, order Connect → Start → Run): the cart stayed still, then **ran into an end stop around the time the model stopped** (~20–21 s). The scope showed x and command ≈ 0 up to 20 s on its ±5 scale.
2. **Sign error by Claude:** from a phone photo of that run's `simout` plot, Claude misread position and command as growing with the **same** sign, concluded the loop was positive feedback, and had the user flip G (numerator `[-2.03 -1.5]`).
   - The same photo also seemed to show the angle creeping by about 0.16 rad without swinging, which is physically implausible for a hanging pendulum. The reading was unreliable, and that run was not exported.
3. **Bumper hit:** on the next attempt the user pressed green Start before Run. **The cart moved sideways before the model was running.** Pressing Run then made it **accelerate into an end stop**, as expected with positive feedback. No damage was reported.
4. **Motor-off check** (drive off, cart pushed by hand by about ±0.2 m) with the flipped build: the printed rows gave command ≈ +1.5x + 2ẋ, confirming the diagram algebra. Together with the acceleration, this proves **+V → +x**, so the **original sign was correct**.
5. **Fix:** the numerator was restored to `[2.03 1.5]`. With Connect → Run → Start, the cart held still for the full run with no drift, even after 20 s. The result of the hand-push part of that test wasn't reported separately. **All 16 recordings use the correct sign.**
6. **Hypothesis (untested):** the Feedback DAC keeps outputting its last value after the real-time model stops, so enabling the drive, or leaving it enabled, with a non-zero last command moves the cart. This would explain both run-aways.
   - Later, Connect + green Start **without** Run left the cart still, because the previous run had ended near 0 V. So Connect → Start → Run was used from run3 onwards.
   - The mitigation used for every recording was to press red Stop after the input ended but before the model's stop time.
   - Next session: open the Feedback DAC block and look for initial/final value parameters.

### 9.5 External-mode logging
The one-shot buffer length decides what reaches `simout`. Setting Duration = T×1000 makes the buffer fill exactly when the input ends, and stop time = T + 5 s leaves time to cut the drive. Whether a partly filled one-shot buffer would be delivered was not tested.

### 9.6 Operator slips (no effect on the data)
- Data set to `u_run6`, which doesn't exist; corrected to `u_run6_square`.
- Export typos (`signals.value`; missing `simout.time`); re-run correctly.
- One free-swing attempt without lifting the pendulum; not exported, and repeated.

### 9.7 `u_run7`
The assignment was missed at first, then typed separately. The data confirm run7 used 0.35 V (§5).

### 9.8 Pendrive virus
The lab PC has a USB "shortcut" virus. The pendrive showed only `HP USB20FD (3868).lnk`, dated 10 Oct 2012 8:21 PM, and the real files were hidden. They were recovered on the user's Mac by showing hidden files. Check the repo for stray files (§1). Recommend that the user reformats the pendrive and tells the lab.

### 9.9 Lab PC clock
It is wrong (Oct 2012), so ignore file modification times. The recording order is in §7.

---

## 10. Open items: ask the user

1. `track_half_length_m`: centre to end stop. Not yet measured when this was written.
2. Pendulum geometry: pivot-to-bob distance; rod and bob masses, if the manual or lab lists them.
3. Mass step: what was attached (hook only, or hook + keys), its mass in grams, its distance from the pivot in cm, and how it was fixed.
4. Whether `obsrec.mdl` was copied off the lab PC. If so, archive it, for example as `scripts/rig/obsrec.mdl`.
5. Whether anything besides the 16 CSVs came off the pendrive.

---

## 11. Tasks for Claude Code, in order

### A. Inventory and integrity
Report a table.

- Confirm the 16 files exist in `data/rig/` (or wherever the user put them) with exactly the names in §7.
- For each file, check:
  - the header;
  - the row count;
  - every `diff(time_s)` is within 1e-9 of 0.001;
  - the value of `time_s[0]` (expected 0; report it);
  - `cart_pos[0] == 0`;
  - `|pend_angle[0] − π| < 1e-4`;
  - min/max agree with §7 to the printed digits.

  A mismatch means a file was corrupted or mixed up in transfer.
- Confirm the quantisation: `cart_pos / (0.078/1024)` and `(pend_angle − π) / (2π/2048)` should be integers to within about 1e-3. If they aren't, report the actual quanta.
- Remove stray pendrive files and don't commit them.

### B. Verify the loop and inputs by reconstructing the excitation
This catches column, sign, amplitude and timing errors.

- Simulink `ode1` is forward Euler. G(s) = 101.5 − 100/(0.02s + 1). With `e_k = −x_k`:
  ```
  y_k     = 101.5*e_k − 100*z_k
  z_{k+1} = 0.95*z_k + 0.05*e_k        (z_0 = 0)
  u_exc_k = command_k − y_k
  ```
- Compare `u_exc` with the nominal input from §5 (amplitude per file) at the logged `time_s`. If the realisation and time alignment are right, expect agreement to about 1e-6 V.
  - If there's a constant lag, estimate it by cross-correlation and report it.
  - A small residual only at encoder count changes means the real-time realisation of G differs slightly; report it, but it isn't a blocker.
  - A large mismatch means a wrong column, sign or amplitude.
- run1 should show command spikes of 101.5 × count that decay to 1.5 × count with τ = 20 ms.
- Also check that the held-out definitions in `rig_inputs.m` (H1–H4: multisine phases, chirp, square, pulse) match the repo's simulation benchmark definitions.

### C. Angle sign
- For any rigid pendulum hanging from the cart, with phi positive when the bob is displaced towards +x, the steady-state gain from cart acceleration to angle is **−1/g ≈ −0.102 rad per m/s²** (from J·phï + m·g·d·phi = −m·d·ẍ).
- Fit a second-order model from ẍ to `phi = pend_angle − π` on run4 and run5. Get ẍ from x with a smoothing differentiator, and use ω_n from run2.
  - DC gain ≈ −0.10 means the logged angle increases when the bob swings towards +x.
  - DC gain ≈ +0.10 means the opposite.
  - The magnitude also confirms the units.
- Map the result to the sign convention the repo's models use, and set `angle_sign` accordingly.

### D. Actuator and friction
- Break-away voltage in each direction from run3, using `command_v` at the onset of motion (for example |Δx| ≥ 3 counts within a pulse). Compare with the assumed +0.15/−0.12 V.
- Estimate Coulomb and viscous friction, and N/V, wherever the repo's actuator model has those parameters.
- Explain the mismatch in §8.3.
- Report before changing any default parameters, and document every change.

### E. Pendulum ω_n and ζ
- From run2 ×3, starting from the first extremum after release.
- Compare the three amplitudes; at ~29° the period should be about θ0²/16 ≈ 1.6% longer.
- Compare ω_n in run7 against run4 trials 2–3 to measure the size of the mass step.

### F. `data/rig/metadata.txt`
Read the metadata parser in `experiments/run_rig_evaluation.py` first, then adapt the draft below to the keys and values it accepts. If it can't express "angle = π when hanging", add explicit support (for example `angle_zero = upright`, or an `angle_offset_rad`) rather than editing the CSVs. The original template's `cart_counts_per_m = 4096` and `angle_counts_per_rev = 4096` don't apply: the data are already in m and rad, and the actual resolutions are 1024 counts per 0.078 m and 2048 counts/rev. Ask the user for the blanks.

```
# units actually logged
cart_units = m
# +1: logged position grows when a positive command pushes the cart (verified)
cart_sign = 1
angle_units = rad
# logged angle = pi with the pendulum hanging at model start (vendor convention, 0 = upright)
angle_zero = upright
# from Task C
angle_sign = __
command_units = V
# manual value, not verified on this rig
newtons_per_volt = 2.0
track_half_length_m = __
pendulum = standard rod, l = __ cm, m = __ g
mass_step_pendulum = __ (what), __ g, taped at __ cm from the pivot
notes = Feedback 33-200/33-201, PCI-1711, MATLAB R2009a RTWT, 1 kHz ode1. Centring loop: Transfer Fcn (2.03s+1.5)/(0.02s+1) on (0 - x), Kp 1.5 V/m, Kd 2.0 V s/m, tau 0.02 s. command_v = total DAC command. A_twotone 0.25 V (run4 trial1), 0.35 V (run4 trials 2-3, run7); A_fading 0.40 V; A_heldout 0.70 V. +x = towards the 'Feedback' logo end. Order: Connect, green Start, Run; red Stop before model stop.
```

### G. Run the evaluation
- First check how it groups trials and picks baselines. **run7 must be compared with run4 trials 2–3** (same 0.35 V input), not trial 1. Check that names without `_trialN` (`run6_heldout_*`, `run7_mass_step_twotone`) are accepted.
- Then run:
  `python experiments/run_rig_evaluation.py --data data/rig --models A B C D E`
  The outputs go to `results/rig/rig/` and `figures/rig/rig/summary.png`.
- Summarise per model, and flag any result driven by the caveats in §8: closed-loop data, near-break-away commands, a small mass step.

### H. Bring the repo in line
- Replace `scripts/rig/rig_inputs.m` and `scripts/rig/export_run.m` with the versions in `as_run_scripts/`. They are the exact code that ran, converted to LF line endings. Document this session's amplitudes in `rig_inputs.m`, and ask the user whether 0.35 should become the default.
- Rewrite README §12.2 (and §12.1 where affected) to the as-run procedure:
  - model built from Cart Control (§3), with the Transfer Fcn loop instead of the vendor PID;
  - set-point slider at 0;
  - Duration rule;
  - rebuild after every input change;
  - Connect → Start → Run, and press red Stop before the model stops;
  - export columns `[1 2 3]`;
  - units m and rad, angle = π when hanging;
  - R2009a constraints;
  - the checksum-mismatch fix;
  - warnings about the lab PC virus and clock.
- Add this handoff to the repo, for example as `docs/rig_session_2026-11-06.md`, alongside the data and metadata.

---

## 12. Files in this bundle

| File | What it is |
|---|---|
| `HANDOFF.md` | This document |
| `as_run_scripts/rig_inputs.m` | R2009a-safe input generator, exactly as run. The A_twotone = 0.35 override was typed at the prompt (§5). |
| `as_run_scripts/export_run.m` | R2009a-safe CSV exporter, exactly as run |

---

## 13. What happened next (added by Claude Code, 6 Nov 2026)

The tasks of §11 were carried out on the same day. Results are in README §12.5; every number there
is reproduced by `python scripts/rig/characterise_rig.py` and
`python experiments/run_rig_evaluation.py --data data/rig --models A B C D E`.

- **A, B.** All 16 files are complete and uncorrupted, and in the folder `data/rig/`. Removing the
  centring loop from `command_v` recovers each run's nominal input to within 3 × 10⁻⁷ V with no lag,
  confirming the columns, the loop's sign and the amplitudes of §5 (0.35 V for run7).
- **C.** The gain from cart acceleration to the logged angle is −0.104 rad per m/s², so
  `angle_sign = -1`. The magnitude is 2 % above 1/g.
- **D.** Break-away: the cart stays at +0.10 V and starts at +0.15 V towards +x, and starts below
  0.10 V towards −x. A cart model fitted while moving gives 3.5 m/s² per V; with the manual's
  2.63 kg that is about 9 N/V, consistent with the manual's ±2.5 V ↔ ±20 N (8 N/V), so
  `newtons_per_volt = 8.0`. The model reproduces the measured cart travel to 5–12 mm RMS, which
  explains §8.3. The vendor DAC block adds its own friction compensation (manual p. 15–16); its
  values on this rig are still unknown.
- **E.** Natural frequency 5.478 rad/s (ζ about 0.001–0.002); the mass step raised it by only
  0.15–0.19 %. The rig carries two pendulum arms fixed to the same shaft, with a short length of rod above the
  pivot (user); the angle shows a single mode. **Correction to §6
  and §10:** the run7 mass was a small stapler taped at the pendulum's bottom weight (user), not a
  key-ring hook mid-rod; it was not weighed. A mass at the bob barely changes the frequency, so run7
  is judged on mass and coupling: the cart-coupling term rises from 0.032–0.035 m (run4 trials 2–3)
  to 0.037 m, within about one trial-to-trial spread.
- **F.** `data/rig/metadata.txt` written; `src/rig/rig_data.py` now mirrors a flipped angle about
  the hanging position, so the rig's "π when hanging" convention is expressed without editing the CSVs.
- **G.** With their simulation settings, the physics-structured models were the worst on the rig.
  Seven things were found in the recordings and added as options that are off in simulation (the
  simulation benchmarks were rerun and are unchanged): the pendulum's real frequency as the gravity
  scale (30 s⁻²; D takes it from the metadata, E learns it online from its pendulum stack and lands
  on 30.0–30.9 s⁻² on the driven and fading runs; E's instantaneous gain follows the scale, 3.3 at
  30); a constant cart force (a push towards −x, as from a track tilted by about 0.7°) and a force
  along the command, k_u ≥ 0, for the DAC block's friction compensation (R² of the cart fit
  0.875 → 0.950), both in B, D and E; leakage of E's cart inertia towards its prior where the cart is
  not excited; an inertia margin of 0.02 for E, because this pendulum's I + ml² ≈ 0.03 lay below the
  simulation's 0.05; no change detection on the rig (false alarms); and, for E, the coupling learned
  from the pendulum's own equation, with only its scale from the cart's (frozen at offline-fitted
  parameters E's observer reaches 0.019–0.025 rad/s on the driven runs, so the learning, not the model,
  had been the limit). Static friction was checked and ruled out. Result (README §12.5.4): E and C
  are tied for the best pendulum-velocity estimate (medians 0.012 and 0.013 rad/s, a gap within the
  velocity reference's own error of about 0.002 rad/s, README §12.4); E is clearly ahead on the mass
  step and C on the low-amplitude two-tone trial 1.
  E predicts the held-out runs about 30 times better than any other model (0.019 against C's 0.66,
  median) and identifies the coupling at the hanging position as 3.11–3.14 m⁻¹ (ω²/g = 3.06), the
  swing frequency as 5.47–5.51 rad/s (measured 5.48), a cart mass of 2.30–2.69 kg (manual 2.63) and a
  compensation of 0.40–0.73 N (default 0.8 N). B is not repaired: its instantaneous law alone cannot
  learn the coupling from these data (E with its stacks off fails the same way); a replay buffer
  gave B's coupling the right sign but not a better error. Every observer's pendulum-velocity
  estimate is far less noisy than the rig's dirty-derivative filter (chattering ratio 1.0–1.6
  against 13–18).
- **H.** `scripts/rig/` holds the as-run MATLAB scripts; README §12.1–§12.2 describe the as-run
  procedure.

- **7 Nov 2026, offline checks (README §12.5.4–§12.5.5).** E's friction residual learns 1–2 % of the
  cart's Coulomb friction and does not help (free swings better only through a torque 10–30 times the
  real damping; driven, staircase and held-out runs no better or worse), so it stays off. The LSTM
  memories of A and C do not learn the friction: their frozen models have no friction step, and C's
  observer is 7–27 % more accurate with its memory switched off. `characterise_rig.py --calibrate`
  writes an automatic calibration (inertia margin, starting friction and force terms, natural
  frequency) into metadata.txt; from one free swing and one two-tone run it reproduces the hand-tuned
  E to within 1 %, while the staircase cannot calibrate the cart (its short pulses misreport the
  acceleration per volt). `data/rig/metadata.txt` has no calibration block yet.

Still open: the stapler's mass and the pendulum's own masses; `obsrec.mdl` was not copied off the lab PC (only a photo of its diagram exists, which matches §3); the
Feedback DAC block's friction-compensation values.
