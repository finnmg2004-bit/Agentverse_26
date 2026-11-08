#!/bin/sh
# Nudge overnight push guard (installed by `python scripts/overnight_check.py --install-hook`).
# Tonight only overnight/* branches may be pushed. Remove in the morning with --uninstall-hook.
while read local_ref local_sha remote_ref remote_sha; do
  case "$remote_ref" in
    refs/heads/overnight/*) ;;
    *)
      echo "[overnight guard] push to '$remote_ref' blocked: tonight only overnight/* branches may be pushed." >&2
      echo "[overnight guard] use: git push origin HEAD:overnight/<lane>" >&2
      exit 1 ;;
  esac
done
exit 0
