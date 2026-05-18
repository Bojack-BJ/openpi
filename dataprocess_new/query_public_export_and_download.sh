#!/usr/bin/env bash

: <<'USAGE'
用途:
  1. 调用 test_query_public_export.py 提交 query_public 导出任务并轮询结果
  2. 下载接口返回的 jsonl 结果文件到当前脚本所在目录
  3. 从 jsonl 中筛选 record_type == "data_item" 的记录
  4. 读取每条记录的 session_path / out_task_id / tos_bucket
  5. 按 out_task_id 去掉最后一个字母后的内容分组
  6. 用 rclone 将云端 session 数据拷贝到本地目标目录
  7. 整批拷贝成功后，自动删除刚下载的 jsonl 文件

目标目录结构:
  /root/Users/dataset/BBB/<out_task_id去掉最后一个字母>/<包目录>/<session目录名>/

常见用法:
  只下载 jsonl，不做 rclone 拷贝:
    ./query_public_export_and_download.sh \
      --task-id 20260407H165Aa \
      --tenant-id 69b001bb14ab30eb5f9dc29d \
      --evaluate-status PASS

  下载 jsonl 后继续分类拷贝:
    RCLONE_PARALLEL_JOBS=48 \
    RCLONE_TRANSFERS=4 \
    RCLONE_CHECKERS=16 \
    RCLONE_MULTI_THREAD_STREAMS=4 \
    ./query_public_export_and_download.sh \
      --copy-name BBB \
      --task-id 20260307K048Ca 20260309K054Aa 20260309K054Ab \
      --tenant-id 69b001bb14ab30eb5f9dc29d \
      --evaluate-status PASS

    RCLONE_PARALLEL_JOBS=48 \
    RCLONE_TRANSFERS=4 \
    RCLONE_CHECKERS=16 \
    RCLONE_MULTI_THREAD_STREAMS=4 \
    ./query_public_export_and_download.sh \
      --copy-name pi0_tasks_lixiaotong/H081A_toy_block_placement \
      --task-id H081 \
      --tenant-id 69b001bb14ab30eb5f9dc29d \
      --evaluate-status PASS

参数说明:
  --copy-name NAME
      开启 rclone 拷贝，并把数据放到 /root/Users/dataset/NAME 下。
      如果不传这个参数，脚本只会下载 jsonl，不会执行 rclone。

  --copy-root DIR
      自定义拷贝根目录，默认 /root/Users/dataset。

  --task-id ...
      支持两种写法:
        --task-id A B C
        --task-id A --task-id B --task-id C

透传参数:
  除脚本自身参数外，其余参数都会原样传给 test_query_public_export.py，
  例如 --tenant-id / --evaluate-status / --base-url / --start-time / --end-time。

性能相关:
  当前脚本默认使用“多 session 并行 + 单个 rclone 内部并发”两层并发。
  可通过修改下面这些环境变量调优:
    RCLONE_PARALLEL_JOBS
    RCLONE_TRANSFERS
    RCLONE_CHECKERS
    RCLONE_MULTI_THREAD_STREAMS
    RCLONE_BUFFER_SIZE
    RCLONE_FAST_LIST
  示例:
    RCLONE_PARALLEL_JOBS=8 RCLONE_TRANSFERS=8 RCLONE_CHECKERS=16 ./query_public_export_and_download.sh ...
USAGE

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
API_SCRIPT="${SCRIPT_DIR}/test_query_public_export.py"
COPY_ROOT_DEFAULT="/root/Users/dataset"
RCLONE_PARALLEL_JOBS="${RCLONE_PARALLEL_JOBS:-4}"
RCLONE_TRANSFERS="${RCLONE_TRANSFERS:-8}"
RCLONE_CHECKERS="${RCLONE_CHECKERS:-16}"
RCLONE_MULTI_THREAD_STREAMS="${RCLONE_MULTI_THREAD_STREAMS:-4}"
RCLONE_BUFFER_SIZE="${RCLONE_BUFFER_SIZE:-16M}"
RCLONE_FAST_LIST="${RCLONE_FAST_LIST:-true}"

COPY_NAME=""
COPY_ROOT="${COPY_ROOT_DEFAULT}"
API_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --copy-name)
      if [[ $# -lt 2 ]]; then
        echo "参数错误: --copy-name 需要一个值。" >&2
        exit 1
      fi
      COPY_NAME="$2"
      shift 2
      ;;
    --copy-root)
      if [[ $# -lt 2 ]]; then
        echo "参数错误: --copy-root 需要一个值。" >&2
        exit 1
      fi
      COPY_ROOT="$2"
      shift 2
      ;;
    --task-id)
      shift
      if [[ $# -eq 0 || "$1" == --* ]]; then
        echo "参数错误: --task-id 后至少需要一个值。" >&2
        exit 1
      fi
      while [[ $# -gt 0 && "$1" != --* ]]; do
        API_ARGS+=("--task-id" "$1")
        shift
      done
      ;;
    -h|--help)
      cat <<EOF
用法:
  $(basename "$0") [脚本自身参数] [test_query_public_export.py 参数]

脚本自身参数:
  --copy-name NAME   下载完成后，把数据拷贝到 ${COPY_ROOT_DEFAULT}/NAME 下并按 out_task_id 分类
  --copy-root DIR    指定拷贝根目录，默认 ${COPY_ROOT_DEFAULT}
  --task-id ...      支持一次传多个 task_id，也兼容重复写多个 --task-id
  -h, --help         显示帮助

其余参数会原样透传给:
  ${API_SCRIPT}
EOF
      exit 0
      ;;
    *)
      API_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ! -f "${API_SCRIPT}" ]]; then
  echo "未找到接口脚本: ${API_SCRIPT}" >&2
  exit 1
fi

TMP_OUTPUT="$(mktemp)"
TMP_DIR="$(mktemp -d)"
cleanup() {
  rm -f "${TMP_OUTPUT}"
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

format_percent() {
  awk -v current="$1" -v total="$2" 'BEGIN { if (total == 0) printf "0.00"; else printf "%.2f", (current * 100) / total }'
}

format_duration() {
  local total_seconds="$1"
  local hours=0
  local minutes=0
  local seconds=0

  if [[ "${total_seconds}" -lt 0 ]]; then
    total_seconds=0
  fi

  hours=$((total_seconds / 3600))
  minutes=$(((total_seconds % 3600) / 60))
  seconds=$((total_seconds % 60))

  if [[ "${hours}" -gt 0 ]]; then
    printf "%02d:%02d:%02d" "${hours}" "${minutes}" "${seconds}"
  else
    printf "%02d:%02d" "${minutes}" "${seconds}"
  fi
}

run_copy_job() {
  local bucket="$1"
  local session_path="$2"
  local target_group_dir="$3"
  shift 3

  mkdir -p "${target_group_dir}"
  rclone copy "volces-tos:${bucket}/${session_path}" "${target_group_dir}" "$@"
}

wait_for_one_copy() {
  local finished_pid=""
  local wait_status=0
  local label=""
  local source_path=""
  local target_path=""
  local log_file=""
  local progress_percent=""
  local elapsed_seconds=0
  local elapsed_text=""
  local eta_seconds=0
  local eta_text="--:--"
  local remaining_pids=()

  if wait -n -p finished_pid "${RUNNING_PIDS[@]}"; then
    wait_status=0
  else
    wait_status=$?
  fi

  COMPLETED_COUNT=$((COMPLETED_COUNT + 1))
  progress_percent="$(format_percent "${COMPLETED_COUNT}" "${TOTAL_COUNT}")"
  elapsed_seconds=$(( $(date +%s) - COPY_START_TS ))
  elapsed_text="$(format_duration "${elapsed_seconds}")"
  if [[ "${COMPLETED_COUNT}" -gt 0 && "${TOTAL_COUNT}" -gt "${COMPLETED_COUNT}" ]]; then
    eta_seconds="$(awk -v elapsed="${elapsed_seconds}" -v done="${COMPLETED_COUNT}" -v total="${TOTAL_COUNT}" 'BEGIN { if (done <= 0) print 0; else printf "%d", (elapsed / done) * (total - done) }')"
    eta_text="$(format_duration "${eta_seconds}")"
  elif [[ "${TOTAL_COUNT}" -eq "${COMPLETED_COUNT}" ]]; then
    eta_text="00:00"
  fi
  label="${PID_LABELS[${finished_pid}]-unknown}"
  source_path="${PID_SOURCES[${finished_pid}]-}"
  target_path="${PID_TARGETS[${finished_pid}]-}"
  log_file="${PID_LOGS[${finished_pid}]-}"

  if [[ "${wait_status}" -eq 0 ]]; then
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    echo "总体进度: ${COMPLETED_COUNT}/${TOTAL_COUNT} (${progress_percent}%)，已耗时: ${elapsed_text}，预计剩余: ${eta_text}，完成: ${label}"
  else
    FAILED_COUNT=$((FAILED_COUNT + 1))
    echo "总体进度: ${COMPLETED_COUNT}/${TOTAL_COUNT} (${progress_percent}%)，已耗时: ${elapsed_text}，预计剩余: ${eta_text}，失败: ${label}" >&2
    {
      echo "===== 失败任务 ${FAILED_COUNT} ====="
      echo "分组/会话: ${label}"
      echo "源路径: ${source_path}"
      echo "目标路径: ${target_path}"
      echo "rclone 退出码: ${wait_status}"
      echo "日志内容:"
      if [[ -n "${log_file}" && -f "${log_file}" ]]; then
        cat "${log_file}"
      else
        echo "(无日志)"
      fi
      echo
    } >> "${FAILED_LOG_PATH}"
  fi

  for pid in "${RUNNING_PIDS[@]}"; do
    if [[ "${pid}" != "${finished_pid}" ]]; then
      remaining_pids+=("${pid}")
    fi
  done
  RUNNING_PIDS=("${remaining_pids[@]}")
  unset PID_LABELS["${finished_pid}"] PID_SOURCES["${finished_pid}"] PID_TARGETS["${finished_pid}"] PID_LOGS["${finished_pid}"]
}

echo "开始提交导出任务并等待结果..."

if ! "${PYTHON_BIN}" "${API_SCRIPT}" "${API_ARGS[@]}" | tee "${TMP_OUTPUT}"; then
  echo "接口调用失败，未执行下载。" >&2
  exit 1
fi

DOWNLOAD_URL="$(
  python3 - "${TMP_OUTPUT}" <<'PY'
import re
import sys

path = sys.argv[1]
text = open(path, "r", encoding="utf-8").read()
matches = re.findall(r"download_url:\s*(https?://\S+)", text)
if matches:
    print(matches[-1])
PY
)"

if [[ -z "${DOWNLOAD_URL}" ]]; then
  echo "未从输出中解析到 download_url，未执行下载。" >&2
  exit 1
fi

FILE_NAME="$(basename "${DOWNLOAD_URL%%\?*}")"
if [[ -z "${FILE_NAME}" || "${FILE_NAME}" == "/" || "${FILE_NAME}" == "." ]]; then
  FILE_NAME="query_public_export.jsonl"
fi

TARGET_PATH="${SCRIPT_DIR}/${FILE_NAME}"

echo "开始下载结果文件..."
echo "下载地址: ${DOWNLOAD_URL}"
echo "保存路径: ${TARGET_PATH}"

curl --fail --location --silent --show-error \
  --output "${TARGET_PATH}" \
  "${DOWNLOAD_URL}"

echo "下载完成: ${TARGET_PATH}"

if [[ -z "${COPY_NAME}" ]]; then
  exit 0
fi

if ! command -v rclone >/dev/null 2>&1; then
  echo "未找到 rclone，无法执行数据拷贝。" >&2
  exit 1
fi

COPY_BASE_DIR="${COPY_ROOT}/${COPY_NAME}"
mkdir -p "${COPY_BASE_DIR}"

echo "开始按 out_task_id 分类拷贝数据..."
echo "目标根目录: ${COPY_BASE_DIR}"
echo "rclone 参数: parallel_jobs=${RCLONE_PARALLEL_JOBS} transfers=${RCLONE_TRANSFERS} checkers=${RCLONE_CHECKERS} multi_thread_streams=${RCLONE_MULTI_THREAD_STREAMS} buffer_size=${RCLONE_BUFFER_SIZE} fast_list=${RCLONE_FAST_LIST}"

COPY_LIST="$(
  "${PYTHON_BIN}" - "${TARGET_PATH}" <<'PY'
import json
import os
import sys

jsonl_path = sys.argv[1]
seen = set()

with open(jsonl_path, "r", encoding="utf-8") as f:
    for raw_line in f:
        line = raw_line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if item.get("record_type") != "data_item":
            continue
        session_path = item.get("session_path")
        out_task_id = item.get("out_task_id")
        bucket = item.get("tos_bucket") or "onestar-tos-vla"
        if not session_path or not out_task_id:
            continue

        group_name = out_task_id[:-1] if len(out_task_id) > 1 else out_task_id
        normalized_path = session_path.rstrip("/")
        session_name = os.path.basename(normalized_path)
        package_name = os.path.basename(os.path.dirname(normalized_path))
        key = (bucket, session_path, group_name, package_name, session_name)
        if key in seen:
            continue
        seen.add(key)
        print("\t".join(key))
PY
)"

if [[ -z "${COPY_LIST}" ]]; then
  echo "JSON 文件中未找到可拷贝的 data_item 记录。" >&2
  exit 1
fi

TOTAL_COUNT="$(printf '%s\n' "${COPY_LIST}" | awk 'NF {count += 1} END {print count + 0}')"
echo "总共需要拷贝 ${TOTAL_COUNT} 条数据路径。"

RCLONE_ARGS=(
  --transfers "${RCLONE_TRANSFERS}"
  --checkers "${RCLONE_CHECKERS}"
  --multi-thread-streams "${RCLONE_MULTI_THREAD_STREAMS}"
  --buffer-size "${RCLONE_BUFFER_SIZE}"
)
if [[ "${RCLONE_FAST_LIST}" == "true" ]]; then
  RCLONE_ARGS+=(--fast-list)
fi

FAILED_LOG_PATH="${SCRIPT_DIR}/query_public_export_failures_$(date +%Y%m%dT%H%M%S).log"
declare -a RUNNING_PIDS=()
declare -A PID_LABELS=()
declare -A PID_SOURCES=()
declare -A PID_TARGETS=()
declare -A PID_LOGS=()

LAUNCHED_COUNT=0
COMPLETED_COUNT=0
SUCCESS_COUNT=0
FAILED_COUNT=0
COPY_START_TS="$(date +%s)"

while IFS=$'\t' read -r BUCKET SESSION_PATH GROUP_NAME PACKAGE_NAME SESSION_NAME; do
  [[ -z "${SESSION_PATH}" ]] && continue
  TARGET_GROUP_DIR="${COPY_BASE_DIR}/${GROUP_NAME}/${PACKAGE_NAME}/${SESSION_NAME}"

  while [[ "${#RUNNING_PIDS[@]}" -ge "${RCLONE_PARALLEL_JOBS}" ]]; do
    wait_for_one_copy
  done

  LAUNCHED_COUNT=$((LAUNCHED_COUNT + 1))
  LOG_FILE="${TMP_DIR}/rclone_copy_${LAUNCHED_COUNT}.log"
  LABEL="${GROUP_NAME}/${PACKAGE_NAME}/${SESSION_NAME}"

  echo "启动拷贝: ${LAUNCHED_COUNT}/${TOTAL_COUNT} -> ${LABEL} (当前并发 ${#RUNNING_PIDS[@]}/${RCLONE_PARALLEL_JOBS})"
  run_copy_job "${BUCKET}" "${SESSION_PATH}" "${TARGET_GROUP_DIR}" "${RCLONE_ARGS[@]}" >"${LOG_FILE}" 2>&1 &
  PID=$!

  RUNNING_PIDS+=("${PID}")
  PID_LABELS["${PID}"]="${LABEL}"
  PID_SOURCES["${PID}"]="${SESSION_PATH}"
  PID_TARGETS["${PID}"]="${TARGET_GROUP_DIR}"
  PID_LOGS["${PID}"]="${LOG_FILE}"
done <<< "${COPY_LIST}"

while [[ "${#RUNNING_PIDS[@]}" -gt 0 ]]; do
  wait_for_one_copy
done

rm -f "${TARGET_PATH}"
echo "已删除下载的结果文件: ${TARGET_PATH}"

if [[ "${FAILED_COUNT}" -gt 0 ]]; then
  echo "分类拷贝结束，成功 ${SUCCESS_COUNT} 条，失败 ${FAILED_COUNT} 条。" >&2
  echo "失败详情已写入: ${FAILED_LOG_PATH}" >&2
  exit 1
fi

rm -f "${FAILED_LOG_PATH}" 2>/dev/null || true
echo "分类拷贝完成，共成功处理 ${SUCCESS_COUNT} 条数据路径。"
