#!/bin/sh
# Report whether each installed ilands-runner release still carries the memorykit section.
MARK="${MK_MARK:-Long-Term Memory (memorykit}"
for f in "$HOME"/.local/share/ilands-runner/releases/*/app/dist/harness/prompt.js; do
  [ -e "$f" ] || { echo "no runner releases found"; exit 1; }
  if grep -q "$MARK" "$f"; then echo "OK      $f"; else echo "MISSING $f"; fi
done
