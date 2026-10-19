#!/usr/bin/env python3
# Vendored unchanged from ~/glm-5.3-reap/scripts/imatrix_verify.py (2026-10-19).
"""Structurally validate a legacy (.dat) imatrix.

llama-imatrix exits 0 even when the final write is short: a failed fwrite/fclose on a full disk
is not checked, so you get rc=0 and a file truncated at a stdio flush boundary. llama-quantize
then dies with 'failed reading data for entry N' -- AFTER the vehicle model has been deleted and
the cheap retry path is gone. So parse it before trusting it.

Exits 0 if every entry the header promises is present and there are no trailing bytes.
"""
import struct, sys, os

def main(path):
    sz = os.path.getsize(path)
    with open(path, 'rb') as f:
        n = struct.unpack('<i', f.read(4))[0]
        if not (0 < n < 1 << 20):
            print(f"FAIL: implausible entry count {n}"); return 1
        for i in range(n):
            hdr = f.read(4)
            if len(hdr) < 4:
                print(f"FAIL: truncated at entry {i}/{n} (name length)"); return 1
            ln = struct.unpack('<i', hdr)[0]
            if not (0 < ln <= 4096):
                print(f"FAIL: entry {i} bogus name length {ln}"); return 1
            name = f.read(ln).decode('utf-8', 'replace')
            meta = f.read(8)
            if len(meta) < 8:
                print(f"FAIL: truncated at entry {i} '{name}' (ncall/nval)"); return 1
            ncall, nval = struct.unpack('<ii', meta)
            need = nval * 4
            if len(f.read(need)) < need:
                print(f"FAIL: truncated at entry {i} '{name}' "
                      f"(ncall={ncall} nval={nval}, file {sz}B)"); return 1
        # The entries are followed by a REQUIRED trailer: [int32 m_last_call][int32 len][dataset].
        # A file with zero trailing bytes is not "clean", it is missing its trailer.
        tail = f.read()
    if len(tail) < 8:
        print(f"FAIL: trailer missing or short ({len(tail)}B) after {n} entries"); return 1
    last_call, dlen = struct.unpack('<ii', tail[:8])
    if dlen != len(tail) - 8:
        print(f"FAIL: trailer dataset length {dlen} != {len(tail)-8} remaining bytes"); return 1
    dataset = tail[8:].decode('utf-8', 'replace')
    print(f"OK: {n} entries, {sz} bytes, {last_call} chunks, dataset={dataset!r}")
    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
