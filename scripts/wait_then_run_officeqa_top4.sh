#!/usr/bin/env bash
# Wait for the officeqa rule-gen grid driver to finish, then launch the officeqa
# top-4 full pipelines. Meant to be launched detached (setsid) so the whole chain
# survives the laptop being off.
#
# The wait condition matches only the python3 grid driver (comm==python3 + its
# cmdline), so it never self-matches this bash wrapper or its ps/awk/grep helpers.

cd "$(dirname "$0")/.."

echo "[wait] $(date '+%F %T') waiting for officeqa rule-gen driver to finish ..."
while ps -eo comm,args | awk '$1=="python3"' | grep -q "run_grid_parallel.py --dataset officeqa"; do
  sleep 60
done
echo "[wait] $(date '+%F %T') officeqa rule-gen driver gone; launching top-4 full pipelines"
exec bash scripts/run_officeqa_top4.sh
