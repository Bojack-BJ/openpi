#!/usr/bin/env bash
'''
Example:
600秒检查一次指定PID是否结束，结束后自动执行pi0_task_Waste_sorting_Aa_qwen3_5.sh脚本。
bash wait_then_run.sh 755448 pi0_task_Waste_sorting_Aa_qwen3_5.sh 600
保存log版（推荐）：
bash /root/Users/lixiaotong/openpi/scripts/wait_then_run.sh 2462510 /root/Users/lixiaotong/openpi/scripts/waste_sorting/pi0_task_Waste_sorting_Aa_qwen3_5.sh 600 2>&1 | tee -a /root/Users/lixiaotong/openpi/scripts/wait_then_run_c.log
后台运行版：
nohup bash /root/Users/lixiaotong/openpi/scripts/wait_then_run.sh <当前训练PID> \
/root/Users/lixiaotong/openpi/scripts/waste_sorting/pi0_task_Waste_sorting_Aa_qwen3_5.sh \
600 > /root/Users/lixiaotong/openpi/scripts/wait_then_run.log 2>&1 &
查询训练PID命令示例：
ps -eo pid,cmd | grep -E "train|pi0_task|openpi" | grep -v grep
'''

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <wait_pid> <next_script_path> [check_interval_seconds]"
  exit 1
fi

WAIT_PID="$1"
NEXT_SCRIPT="$2"
INTERVAL="${3:-15}"

if ! [[ "$WAIT_PID" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] wait_pid must be an integer, got: $WAIT_PID"
  exit 1
fi

if [[ ! -f "$NEXT_SCRIPT" ]]; then
  echo "[ERROR] next script not found: $NEXT_SCRIPT"
  exit 1
fi

if [[ ! -x "$NEXT_SCRIPT" ]]; then
  echo "[WARN] next script is not executable, trying: chmod +x $NEXT_SCRIPT"
  chmod +x "$NEXT_SCRIPT"
fi

if [[ ! -r "/proc/$WAIT_PID/stat" ]]; then
  echo "[ERROR] PID $WAIT_PID does not exist now. Please confirm the running training PID."
  exit 1
fi

# /proc/<pid>/stat field 22 is start time (in clock ticks), used to avoid PID reuse confusion.
ORIGINAL_START_TIME="$(awk '{print $22}' "/proc/$WAIT_PID/stat")"

echo "[$(date '+%F %T')] Watching PID $WAIT_PID"
echo "[$(date '+%F %T')] Next script: $NEXT_SCRIPT"
echo "[$(date '+%F %T')] Check interval: ${INTERVAL}s"

while true; do
  if [[ ! -r "/proc/$WAIT_PID/stat" ]]; then
    echo "[$(date '+%F %T')] PID $WAIT_PID has exited."
    break
  fi

  CURRENT_START_TIME="$(awk '{print $22}' "/proc/$WAIT_PID/stat")"
  if [[ "$CURRENT_START_TIME" != "$ORIGINAL_START_TIME" ]]; then
    echo "[$(date '+%F %T')] PID $WAIT_PID was reused. Original process has ended."
    break
  fi

  sleep "$INTERVAL"
done

echo "[$(date '+%F %T')] Launching: $NEXT_SCRIPT"
bash "$NEXT_SCRIPT"
EXIT_CODE=$?
echo "[$(date '+%F %T')] Next script finished with exit code: $EXIT_CODE"
exit "$EXIT_CODE"
