#!/usr/bin/env python3
import importlib
import importlib.metadata as md
import os
import sys
import traceback

def print_dist_info(dist):
    name = dist.metadata.get("Name") or dist.metadata.get("name") or "<unknown>"
    print(f"\nDistribution: {name}  version: {dist.version}")
    try:
        loc = dist.locate_file('')
        print("  location:", str(loc))
    except Exception:
        pass
    try:
        txt = dist.read_text("direct_url.json")
        if txt:
            print("  direct_url.json:")
            print(txt.strip())
        else:
            print("  direct_url.json: (empty)")
    except FileNotFoundError:
        print("  direct_url.json: NOT FOUND")
    except Exception as e:
        print("  direct_url.json: error:", e)
    try:
        files = list(dist.files or [])[:12]
        if files:
            print("  sample files:")
            for f in files:
                print("   -", f)
    except Exception:
        pass

def main():
    print("Looking for installed distributions with 'libero' in the name...")
    found = []
    try:
        for d in md.distributions():
            nm = (d.metadata.get("Name") or "").lower()
            if "libero" in nm:
                found.append(d)
    except Exception:
        print("Failed to list distributions:", traceback.format_exc())
        sys.exit(2)

    if not found:
        print("No distribution with 'libero' in the name found.")
    else:
        for d in found:
            print_dist_info(d)

    print("\nTrying to import module 'libero' to locate source files...")
    try:
        mod = importlib.import_module("libero")
        path = getattr(mod, "__file__", None)
        print("libero module file:", path)
        root = os.path.dirname(path) if path else None
        if root:
            # look up a few levels for a .git directory (editable install)
            cur = root
            found_git = False
            for _ in range(6):
                if os.path.isdir(os.path.join(cur, ".git")):
                    print("Found .git directory at:", os.path.join(cur, ".git"))
                    found_git = True
                    break
                cur = os.path.dirname(cur)
            if not found_git:
                print(".git directory not found near module path (likely wheel/site-packages install).")
    except Exception as e:
        print("Could not import 'libero':", e)

    print("\nInterpretation hints:")
    print("- If any direct_url.json contains a URL starting with 'git+https://github.com/stepanfeduniak/lerobot-libero-pro', you installed LIBERO-PRO from that repo.")
    print("- If direct_url.json points to hf-libero or to a PyPI URL, you have the original LIBERO package.")
    print("- If there's a .git directory near the module path, you likely installed an editable/git checkout.")
    print("\nDone.")

if __name__ == '__main__':
    main()
