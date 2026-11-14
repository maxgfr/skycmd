"""Runs inside the CredData container only (issues.py obfuscate starts it, without a network):
CredData's own `get_obfuscated_value`, from the files of its checkout mounted at /creddata, on a
list of values (docs/s6-secrets2-prereg.md, section 7.4, obfuscation sensitivity).

    python obfuscate_values.py /creddata /work SEED

/work/values.json comes in (a list of strings), /work/obfuscated.json goes out (a list of the
same length; null where the function failed, did not end in two seconds, or changed the length).
Both hold values: /work is in the data folder and is emptied by the caller.

An issue-report value has no meta row. The function reads two fields of one: PredefinedPattern
("Info" means no obfuscation) and Category ("Basic Authorization" has its own branch). Both are
given empty, so only the branches that read the value's own shape apply. The random generator
is seeded per value with SEED xor the value's index, as CredData seeds it per row.
"""
import json
import os
import random
import signal
import sys
import types


class TooLong(Exception):
    pass


def alarm(*_):
    raise TooLong()


def main():
    creddata, work, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
    sys.path.insert(0, creddata)
    import obfuscate_creds

    row = types.SimpleNamespace(PredefinedPattern="", Category="")
    with open(os.path.join(work, "values.json")) as f:
        values = json.load(f)
    out, failed = [], {"error": 0, "timeout": 0, "length": 0}
    signal.signal(signal.SIGALRM, alarm)
    for i, v in enumerate(values):
        random.seed(seed ^ i)
        signal.alarm(2)
        try:
            new = obfuscate_creds.get_obfuscated_value(v, row)
        except TooLong:
            new = None
            failed["timeout"] += 1
        except Exception:
            new = None
            failed["error"] += 1
        finally:
            signal.alarm(0)
        if new is not None and len(new) != len(v):
            new = None
            failed["length"] += 1
        out.append(new)
    with open(os.path.join(work, "obfuscated.json"), "w") as f:
        json.dump(out, f)
    print(json.dumps({"values": len(values), "unchanged": sum(1 for a, b in zip(values, out) if a == b), **failed}))


if __name__ == "__main__":
    main()
