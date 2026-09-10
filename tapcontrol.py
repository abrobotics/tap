#!/usr/bin/env python3

import subprocess
import time

import numpy as np


DEVICE = "hw:0,0"
RATE = 48000
CHANNELS = 2
PERIOD = 512

# Settings that can be adjusted after the first test.
CALIBRATION_SECONDS = 2.5
MIN_THRESHOLD = 1200
THRESHOLD_MULTIPLIER = 1.8
REFRACTORY = 0.25  # Prevents the same tap from being detected multiple times.
MAX_LAG = 24  # ±24 samples = ±500 µs at 48 kHz.
WINDOW = 240  # 5 ms of signal used to estimate the delay.
MIN_CORRELATION = 0.45


def estimate_lag(left, right, max_lag=80):
    """Find the lag that maximizes correlation between both signals."""
    left = left.astype(np.float64)
    right = right.astype(np.float64)

    left -= np.mean(left)
    right -= np.mean(right)

    best_lag = 0
    best_score = -1.0

    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            a = left[-lag:]
            b = right[:lag]
        elif lag > 0:
            a = left[:-lag]
            b = right[lag:]
        else:
            a = left
            b = right

        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            continue

        score = abs(np.dot(a, b) / denom)
        if score > best_score:
            best_score = score
            best_lag = lag

    return best_lag, best_score


def classify(lag, correlation):
    """Approximately classify a tap using its inter-channel delay."""
    if correlation < MIN_CORRELATION or abs(lag) >= MAX_LAG:
        return "UNRELIABLE SIGNAL"

    # Real-world tests show that the physical channel direction is reversed
    # compared with the initial assumption.
    if lag >= 3:
        return "LEFT"

    if lag <= -3:
        return "RIGHT"

    # Amplitude is not used for classification because the right channel has
    # noticeably higher gain regardless of the tapped area.
    return "CENTER"


arecord = subprocess.Popen(
    [
        "arecord",
        "-q",
        "-D",
        DEVICE,
        "-t",
        "raw",
        "-f",
        "S16_LE",
        "-r",
        str(RATE),
        "-c",
        str(CHANNELS),
        "--period-size",
        str(PERIOD),
    ],
    stdout=subprocess.PIPE,
)

if arecord.stdout is None:
    raise RuntimeError("Unable to open the arecord audio stream.")

print()
print("Experimental TapControl")
print("========================")
print(f"Device     : {DEVICE}")
print(f"Sample rate: {RATE} Hz")
print(f"Resolution : {1_000_000 / RATE:.2f} µs per sample")
print()
print("Press Ctrl+C to quit.")
print()

try:
    print(f"Calibrating for {CALIBRATION_SECONDS:.1f} s: do not tap...")

    calibration_peaks = []
    previous_sample = np.zeros((1, CHANNELS), dtype=np.int32)
    calibration_end = time.monotonic() + CALIBRATION_SECONDS

    while time.monotonic() < calibration_end:
        data = arecord.stdout.read(PERIOD * CHANNELS * np.dtype(np.int16).itemsize)
        if not data:
            raise RuntimeError("arecord stopped during calibration.")

        samples = np.frombuffer(data, dtype=np.int16)
        if len(samples) % CHANNELS:
            continue

        samples = samples.reshape(-1, CHANNELS).astype(np.int32)
        high_passed = np.diff(np.vstack((previous_sample, samples)), axis=0)
        previous_sample = samples[-1:]
        calibration_peaks.append(float(np.max(np.abs(high_passed))))

    noise_p99 = float(np.percentile(calibration_peaks, 99))
    threshold = max(MIN_THRESHOLD, noise_p99 * THRESHOLD_MULTIPLIER)
    print(f"Transient noise p99: {noise_p99:.0f}")
    print(f"Detection threshold: {threshold:.0f}")
    print("Ready — tap at different locations around the computer.\n")

    last_tap = 0.0
    armed = True

    # The previous block provides the signal preceding the tap onset.
    previous = np.zeros((PERIOD, CHANNELS), dtype=np.int16)

    while True:
        data = arecord.stdout.read(PERIOD * CHANNELS * np.dtype(np.int16).itemsize)

        if not data:
            return_code = arecord.poll()
            raise RuntimeError(
                f"arecord stopped unexpectedly (exit code {return_code})."
            )

        samples = np.frombuffer(data, dtype=np.int16)
        if len(samples) % CHANNELS:
            continue

        samples = samples.reshape(-1, CHANNELS)

        # A simple derivative attenuates background noise and emphasizes the
        # short attack that characterizes a tap.
        joined = np.vstack((previous, samples)).astype(np.int32)
        transient = np.diff(joined, axis=0)
        current_transient = transient[-len(samples) :]
        block_peak = float(np.max(np.abs(current_transient)))

        now = time.monotonic()

        if block_peak < threshold * 0.55:
            armed = True

        if (
            armed
            and block_peak >= threshold
            and now - last_tap >= REFRACTORY
        ):
            armed = False
            last_tap = now

            energy = np.maximum(
                np.abs(transient[:, 0]),
                np.abs(transient[:, 1]),
            )
            peak_index = int(np.argmax(energy))

            start = max(0, peak_index - WINDOW // 4)
            end = min(len(transient), peak_index + WINDOW)
            segment = transient[start:end]

            if len(segment) > MAX_LAG * 2 + 20:
                seg_l = segment[:, 0]
                seg_r = segment[:, 1]
                lag, corr = estimate_lag(seg_l, seg_r, MAX_LAG)

                peak_l = int(np.max(np.abs(seg_l.astype(np.int32))))
                peak_r = int(np.max(np.abs(seg_r.astype(np.int32))))
                lag_us = lag / RATE * 1_000_000
                location = classify(lag, corr)

                print(
                    f"TAP  {location:20s} | "
                    f"L={peak_l:5d} R={peak_r:5d} | "
                    f"lag={lag:+4d} samples ({lag_us:+7.1f} µs) | "
                    f"corr={corr:.3f}"
                )

        previous = samples.copy()

except KeyboardInterrupt:
    print("\nStopped.")

finally:
    arecord.stdout.close()
    if arecord.poll() is None:
        arecord.terminate()
        try:
            arecord.wait(timeout=1)
        except subprocess.TimeoutExpired:
            arecord.kill()
            arecord.wait()
