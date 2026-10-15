"""Read HWiNFO sensor values through its shared-memory interface (Windows).

HWiNFO must be running with "Shared Memory Support" enabled and the sensor
window open. The free edition turns the option off after 12 hours.

    python hwinfo_sm.py list [--type power]
    python hwinfo_sm.py idle <seconds> --match "<label substring>" [--match ...]
    python hwinfo_sm.py run --match "<label substring>" [--match ...] -- <command ...>
    add --steady "<label substring>>=<value>" to also summarize only the samples
    where that reading is at least <value> (for example the NPU's usage), which
    excludes host-side setup before the device is busy

`idle` and `run` sample the matched readings every --interval seconds (HWiNFO's
own polling period bounds how often values change) and report the mean, min
and max of each in its own unit, plus energy (the trapezoid integral) for
readings in watts. `run` also
prints the command's own output and wall time. --csv writes every sample.

While `idle` or `run` is active the process asks Windows to stay awake with the
display on (SetThreadExecutionState), so Modern Standby cannot suspend a
measurement; Windows drops the request when the process exits. Nothing in the
user's power settings is changed.
"""

import argparse
import ctypes
import ctypes.wintypes as wt
import struct
import subprocess
import sys
import threading
import time

FILE_MAP_READ = 0x0004
MAP_NAME = "Global\\HWiNFO_SENS_SM2"
READING_TYPES = {0: "none", 1: "temp", 2: "volt", 3: "fan", 4: "current", 5: "power", 6: "clock", 7: "usage", 8: "other"}

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
k32.OpenFileMappingW.restype = wt.HANDLE
k32.OpenFileMappingW.argtypes = [wt.DWORD, wt.BOOL, wt.LPCWSTR]
k32.MapViewOfFile.restype = ctypes.c_void_p
k32.MapViewOfFile.argtypes = [wt.HANDLE, wt.DWORD, wt.DWORD, wt.DWORD, ctypes.c_size_t]
k32.UnmapViewOfFile.argtypes = [ctypes.c_void_p]
k32.CloseHandle.argtypes = [wt.HANDLE]
k32.SetThreadExecutionState.restype = ctypes.c_uint32
k32.SetThreadExecutionState.argtypes = [ctypes.c_uint32]
ES_CONTINUOUS, ES_SYSTEM_REQUIRED, ES_DISPLAY_REQUIRED = 0x80000000, 0x00000001, 0x00000002


def keep_awake(on):
    flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED if on else 0)
    if k32.SetThreadExecutionState(flags) == 0:
        print("warning: SetThreadExecutionState failed; the system may sleep mid-run", file=sys.stderr)


def _cstr(b):
    return b.split(b"\0", 1)[0].decode("latin-1", "replace").strip()


class SharedMemory:
    def __init__(self):
        self.h = k32.OpenFileMappingW(FILE_MAP_READ, False, MAP_NAME)
        if not self.h:
            sys.exit(
                f"cannot open {MAP_NAME} (error {ctypes.get_last_error()}): is HWiNFO running "
                "with Shared Memory Support enabled and the sensor window open?"
            )
        self.view = k32.MapViewOfFile(self.h, FILE_MAP_READ, 0, 0, 0)
        if not self.view:
            sys.exit(f"MapViewOfFile failed (error {ctypes.get_last_error()})")
        hdr = ctypes.string_at(self.view, 44)
        sig, ver, rev, poll_time, so, ss, sn, ro, rs, rn = struct.unpack_from("<IIIqIIIIII", hdr)
        if sig != 0x53695748:  # "HWiS"
            sys.exit(f"unexpected signature {sig:#x}; HWiNFO shared memory may be disabled")
        self.version, self.revision = ver, rev
        self.sensor_off, self.sensor_size, self.n_sensors = so, ss, sn
        self.reading_off, self.reading_size, self.n_readings = ro, rs, rn

    def sensors(self):
        out = []
        for i in range(self.n_sensors):
            b = ctypes.string_at(self.view + self.sensor_off + i * self.sensor_size, self.sensor_size)
            sid, inst = struct.unpack_from("<II", b)
            out.append((sid, inst, _cstr(b[8:136]), _cstr(b[136:264])))
        return out

    def readings(self):
        """(type, sensor_index, reading_id, label, unit, value, min, max, avg) per reading."""
        out = []
        for i in range(self.n_readings):
            b = ctypes.string_at(self.view + self.reading_off + i * self.reading_size, self.reading_size)
            rtype, sidx, rid = struct.unpack_from("<III", b)
            label_user = _cstr(b[140:268]) or _cstr(b[12:140])
            unit = _cstr(b[268:284])
            val, vmin, vmax, vavg = struct.unpack_from("<dddd", b, 284)
            out.append((rtype, sidx, rid, label_user, unit, val, vmin, vmax, vavg))
        return out

    def close(self):
        k32.UnmapViewOfFile(self.view)
        k32.CloseHandle(self.h)


def sensor_name(sensors, idx):
    """Display name of a sensor: the user-set name if any, else the original."""
    return sensors[idx][3] or sensors[idx][2]


def select(sm, patterns):
    sensors = sm.sensors()
    readings = sm.readings()
    chosen = []
    for pat in patterns:
        hits = [(i, r) for i, r in enumerate(readings) if pat.lower() in r[3].lower()]
        if not hits:
            sys.exit(f"no reading label contains {pat!r}; run `list` to see them")
        if len(hits) > 1:
            names = "; ".join(f"[{sensor_name(sensors, r[1])}] {r[3]}" for _, r in hits)
            sys.exit(f"{pat!r} matches {len(hits)} readings, be more specific: {names}")
        i, r = hits[0]
        chosen.append((i, f"[{sensor_name(sensors, r[1])}] {r[3]}", r[4]))
    return chosen


def sample(sm, chosen, interval, stop, rows):
    t0 = time.perf_counter()
    while not stop.is_set():
        rd = sm.readings()
        rows.append((time.perf_counter() - t0, [rd[i][5] for i, _, _ in chosen]))
        stop.wait(interval)
    rd = sm.readings()
    rows.append((time.perf_counter() - t0, [rd[i][5] for i, _, _ in chosen]))


def print_stats(chosen, rows, prefix="", with_energy=True):
    for k, (_, name, unit) in enumerate(chosen):
        vals = [r[1][k] for r in rows]
        line = f"{prefix}{name}: mean {sum(vals) / len(vals):.2f}  min {min(vals):.2f}  max {max(vals):.2f} {unit}"
        if with_energy and unit.upper() == "W":
            energy = sum((rows[j + 1][0] - rows[j][0]) * (vals[j] + vals[j + 1]) / 2 for j in range(len(rows) - 1))
            line += f"  energy {energy:.2f} J"
        print(line)


def summarize(chosen, rows, csv_path, steady=None):
    print(f"samples: {len(rows)} over {rows[-1][0] - rows[0][0]:.2f} s")
    print_stats(chosen, rows)
    if steady:
        k, threshold = steady
        sub = [r for r in rows if r[1][k] >= threshold]
        if sub:
            print(f"steady ({chosen[k][1]} >= {threshold:g}): {len(sub)} samples, {sub[0][0]:.1f} to {sub[-1][0]:.1f} s")
            # No energy here: the subset can have gaps, and integrating across
            # them would count the non-steady time as steady.
            print_stats(chosen, sub, prefix="steady ", with_energy=False)
        else:
            print(f"steady: no sample has {chosen[k][1]} >= {threshold:g}")
    if csv_path:
        with open(csv_path, "w") as f:
            f.write("t_s," + ",".join(f'"{n}"' for _, n, _ in chosen) + "\n")
            for t, v in rows:
                f.write(f"{t:.4f}," + ",".join(f"{x:.4f}" for x in v) + "\n")


def main():
    argv = sys.argv[1:]
    cmd = []
    if "--" in argv:
        cmd = argv[argv.index("--") + 1 :]
        argv = argv[: argv.index("--")]
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["list", "idle", "run"])
    p.add_argument("seconds", nargs="?", type=float, default=60.0)
    p.add_argument("--type", help="list only: reading type filter (power, temp, clock, ...)")
    p.add_argument("--match", action="append", default=[])
    p.add_argument("--interval", type=float, default=0.1)
    p.add_argument("--csv")
    p.add_argument("--steady", help='idle/run: "<label substring>>=<value>", also summarize only those samples')
    a = p.parse_args(argv)
    if a.mode == "run" and not cmd:
        sys.exit("run needs a command after --")
    steady_label = steady_value = None
    if a.steady:
        if ">=" not in a.steady:
            sys.exit('--steady must look like "<label substring>>=<value>"')
        steady_label, v = a.steady.rsplit(">=", 1)
        steady_value = float(v)
        if steady_label not in a.match:
            a.match.append(steady_label)

    sm = SharedMemory()
    if a.mode == "list":
        sensors = sm.sensors()
        print(f"HWiNFO shared memory v{sm.version}.{sm.revision}: {sm.n_sensors} sensors, {sm.n_readings} readings")
        for rtype, sidx, rid, label, unit, val, *_ in sm.readings():
            t = READING_TYPES.get(rtype, str(rtype))
            if a.type and t != a.type:
                continue
            print(f"{t:8} [{sensor_name(sensors, sidx)}] {label} = {val:.3f} {unit}")
        return
    if not a.match:
        sys.exit("--match is required for idle and run")
    chosen = select(sm, a.match)
    steady = (a.match.index(steady_label), steady_value) if a.steady else None
    keep_awake(True)
    rows, stop = [], threading.Event()
    th = threading.Thread(target=sample, args=(sm, chosen, a.interval, stop, rows), daemon=True)
    th.start()
    rc = 0
    try:
        if a.mode == "idle":
            time.sleep(a.seconds)
        else:
            t0 = time.perf_counter()
            try:
                rc = subprocess.call(cmd)
            except OSError as e:
                print(f"cannot run {cmd[0]}: {e}", file=sys.stderr)
                rc = 127
            print(f"command wall time: {time.perf_counter() - t0:.3f} s, exit {rc}")
    finally:
        stop.set()
        th.join()
        keep_awake(False)
    summarize(chosen, rows, a.csv, steady)
    sm.close()
    sys.exit(rc)


if __name__ == "__main__":
    main()
