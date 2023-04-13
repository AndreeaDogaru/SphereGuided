#!/bin/bash

METHOD="$1"
shift

case "$METHOD" in
  neus|nwarp|unisurf)
    "./scripts/extract_$METHOD.sh" "$@"
    ;;
  *)
    echo "Invalid method: $METHOD"
    exit 1
    ;;
esac
