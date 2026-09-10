# TapControl

TapControl detects taps around a laptop with its two built-in microphones and
classifies them as `LEFT`, `CENTER`, or `RIGHT`. It runs on Linux through ALSA
and estimates the arrival-time difference between both audio channels.

## Requirements

- Python 3
- NumPy
- ALSA utilities (`arecord`)
- A stereo microphone input

On Ubuntu:

```bash
sudo apt install python3-numpy alsa-utils
```

## Run

```bash
./tapcontrol.py
```

Keep the area quiet and do not tap during the initial 2.5-second calibration.
After `Ready` appears, tap to the left, center, or right of the computer. Stop
the program with `Ctrl+C`.

Example:

```text
TAP  LEFT                 | L=4539 R=6547 | lag=  +5 samples | corr=0.578
TAP  CENTER               | L=4769 R=8544 | lag=  +0 samples | corr=0.784
TAP  RIGHT                | L=4577 R=7133 | lag=  -4 samples | corr=0.801
```

`UNRELIABLE SIGNAL` means the two channels were not correlated strongly enough
for a trustworthy classification.

## Audio device

The default input is `hw:0,0`. List available capture devices with:

```bash
arecord -l
```

If necessary, change `DEVICE` near the top of `tapcontrol.py`.
