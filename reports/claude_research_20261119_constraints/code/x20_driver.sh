#!/bin/bash
# x20 -> x02 (CNN, 3 polling slots) -> z13 scoring. Own folder only, CPU only, nice 19.
# Memory guard: extraction workers are paused (SIGSTOP) while MemAvailable < 15 GB and resumed at >= 18 GB.
W=/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints; cd $W || exit 1
AV=/data_pool/khangle/facevitals/envs/storage_pilot/bin/python; PY=/data_pool/khangle/facevitals/envs/hr/bin/python
CONDS=std:c4_qvga,std:c6_us1,aidtrack:c1_g25gv01,std:c2_vit,track:c1_g25p,track:c7_blue,std:c1_g50p,std:c1_g12p,std:c1_g25gv001,std:c7_white
echo "$(date '+%F %T') start" >> logs/x20_driver.log
nice -n 19 $AV code/x20_prefilter_conditions.py ${1:-9} $CONDS > logs/x20_run.log 2>&1 &
XP=$!
SL=""
for s in 0 1 2; do nice -n 19 $PY code/x02_infer.py $s 3 x20 > logs/x02_x20_$s.log 2>&1 & SL="$SL $!"; done
while kill -0 $XP 2>/dev/null; do
  a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
  if [ "$a" -lt 15 ]; then pkill -STOP -P $XP; echo "$(date '+%F %T') paused, MemAvailable ${a} GB" >> logs/x20_driver.log
  elif [ "$a" -ge 18 ]; then pkill -CONT -P $XP; fi
  sleep 20
done
echo "$(date '+%F %T') extraction finished" >> logs/x20_driver.log
wait $SL
echo "$(date '+%F %T') CNN inference finished" >> logs/x20_driver.log
nice -n 19 $PY code/z13_public_p4_prefilter.py > logs/z13.log 2>&1
echo "$(date '+%F %T') z13 finished rc=$?" >> logs/x20_driver.log
date '+%F %T' > out/z13_DONE
