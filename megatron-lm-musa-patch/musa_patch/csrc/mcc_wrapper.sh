#!/usr/bin/env bash
set -euo pipefail

args=()
while (($#)); do
  if [[ "$1" == "--compiler-options" ]]; then
    shift
    (($#)) || {
      echo "missing value for --compiler-options" >&2
      exit 2
    }
    args+=("$1")
  else
    args+=("$1")
  fi
  shift
done

musa_root=${MUSA_HOME:-${MUSA_PATH:-/usr/local/musa}}
exec "${musa_root}/bin/mcc" "${args[@]}"
