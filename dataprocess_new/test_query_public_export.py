#!/usr/bin/env python3
import argparse
import json
import sys
import time
from typing import Any, Dict, Optional

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="测试 query_public 异步导出：提交导出任务 -> 轮询查询导出状态"
    )
    parser.add_argument(
        "--base-url",
        default="http://192.168.32.64:8001/api/v1",
        help="API 基础地址，默认 http://127.0.0.1:8001/api/v1",
    )
    parser.add_argument(
        "--tenant-id",
        required=True,
        help="租户ID，会以 query 参数传给 query_public",
    )
    parser.add_argument(
        "--task-id",
        action="append",
        default=None,
        help="任务外部ID，可传多次，例如 --task-id 20260303O043 --task-id 20260303O044",
    )
    parser.add_argument(
        "--scene-type",
        action="append",
        default=None,
        help="场景类型，可传多次，例如 --scene-type O",
    )
    parser.add_argument(
        "--data-source",
        action="append",
        default=None,
        help="数据来源，可传多次，例如 --data-source data_factory --data-source in_the_wild",
    )
    parser.add_argument(
        "--gripper-distribution",
        action="append",
        default=None,
        help="设备类型筛选，可传多次，例如 --gripper-distribution Portable",
    )
    parser.add_argument(
        "--project-distribution",
        action="append",
        default=None,
        help="项目名称筛选，可传多次，例如 --project-distribution FastUMI工厂项目A",
    )
    parser.add_argument(
        "--evaluate-status",
        action="append",
        default=None,
        help="评估结果筛选，可传多次，例如 --evaluate-status PASS",
    )
    parser.add_argument(
        "--keyword",
        default=None,
        help="关键字筛选",
    )
    parser.add_argument(
        "--start-time",
        default=None,
        help="时间范围开始，例如 2026-03-19T00:00:00.000Z",
    )
    parser.add_argument(
        "--end-time",
        default=None,
        help="时间范围结束，例如 2026-03-23T23:59:59.000Z",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="轮询间隔秒数，默认 2",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="轮询超时时间，单位秒，默认 600",
    )
    parser.add_argument(
        "--submit-only",
        action="store_true",
        help="只提交导出任务，不轮询状态",
    )
    parser.add_argument(
        "--empty-body",
        action="store_true",
        help="强制提交空请求体 {}，忽略其他查询参数",
    )
    return parser.parse_args()


def ensure_ok(resp: requests.Response, step: str) -> Dict[str, Any]:
    try:
        payload = resp.json()
    except Exception as exc:
        raise RuntimeError(
            f"{step} 返回非 JSON，status={resp.status_code} body={resp.text}"
        ) from exc

    if resp.status_code >= 400:
        raise RuntimeError(
            f"{step} HTTP失败，status={resp.status_code} body={json.dumps(payload, ensure_ascii=False)}"
        )
    if payload.get("code") != 0:
        raise RuntimeError(
            f"{step} 业务失败，code={payload.get('code')} msg={payload.get('msg')} payload={json.dumps(payload, ensure_ascii=False)}"
        )
    data = payload.get("data")
    if data is None:
        raise RuntimeError(
            f"{step} 成功但 data 为空: {json.dumps(payload, ensure_ascii=False)}"
        )
    return data


def build_request_body(args: argparse.Namespace) -> Dict[str, Any]:
    if args.empty_body:
        return {}

    body: Dict[str, Any] = {}
    if args.keyword:
        body["keyword_query"] = args.keyword
    if args.task_id:
        body["task_id_query"] = args.task_id
    if args.data_source:
        body["data_source_query"] = args.data_source
    if args.gripper_distribution:
        body["gripper_distribution"] = args.gripper_distribution
    if args.project_distribution:
        body["project_distribution"] = args.project_distribution
    if args.scene_type:
        body["scene_type_query"] = args.scene_type
    if args.evaluate_status:
        body["evaluate_status_query"] = args.evaluate_status
    if args.start_time or args.end_time:
        time_range = []
        if args.start_time:
            time_range.append(args.start_time)
        if args.end_time:
            time_range.append(args.end_time)
        body["time_range_query"] = time_range
    return body


def submit_export(
    base_url: str, tenant_id: Optional[str], body: Dict[str, Any]
) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    if tenant_id:
        params["tenant_id"] = tenant_id

    resp = requests.post(
        f"{base_url.rstrip('/')}/data/query_public",
        params=params,
        json=body,
        timeout=60,
    )
    return ensure_ok(resp, "提交导出任务")


def get_record(base_url: str, record_id: str) -> Dict[str, Any]:
    resp = requests.get(
        f"{base_url.rstrip('/')}/data/query_public/{record_id}",
        timeout=30,
    )
    return ensure_ok(resp, "查询导出记录")


def print_submit_result(data: Dict[str, Any]) -> None:
    print("导出任务已提交:")
    print(f"  record_id: {data.get('record_id')}")
    print(f"  status: {data.get('status')}")
    print(f"  bucket: {data.get('bucket')}")
    print(f"  object_key: {data.get('object_key')}")
    print(f"  download_url: {data.get('download_url')}")
    if data.get("message"):
        print(f"  message: {data.get('message')}")


def print_record_status(record: Dict[str, Any]) -> None:
    print(
        "状态更新:"
        f" status={record.get('status')}"
        f" exported_count={record.get('exported_count')}"
        f" total_records={record.get('total_records')}"
        f" export_elapsed_seconds={record.get('export_elapsed_seconds')}"
        f" upload_elapsed_seconds={record.get('upload_elapsed_seconds')}"
        f" total_elapsed_seconds={record.get('total_elapsed_seconds')}"
    )
    if record.get("message"):
        print(f"  message: {record.get('message')}")
    if record.get("error"):
        print(f"  error: {record.get('error')}")


def main() -> int:
    args = parse_args()
    body = build_request_body(args)

    print("提交导出任务中...")
    print(f"请求体: {json.dumps(body, ensure_ascii=False)}")
    submitted = submit_export(args.base_url, args.tenant_id, body)
    print_submit_result(submitted)

    if args.submit_only:
        return 0

    record_id = submitted["record_id"]
    deadline = time.time() + args.timeout
    last_status = None
    last_exported_count = None
    last_message = None

    while time.time() < deadline:
        record = get_record(args.base_url, record_id)
        status = record.get("status")
        exported_count = record.get("exported_count")
        message = record.get("message")
        if (
            status != last_status
            or exported_count != last_exported_count
            or message != last_message
            or status in {"FAILED", "SUCCESS"}
        ):
            print_record_status(record)
            last_status = status
            last_exported_count = exported_count
            last_message = message

        if status == "SUCCESS":
            print("导出成功，可下载结果文件:")
            print(f"  download_url: {record.get('download_url')}")
            print(f"  bucket: {record.get('bucket')}")
            print(f"  object_key: {record.get('object_key')}")
            return 0

        if status == "FAILED":
            print("导出失败。")
            return 1

        time.sleep(args.poll_interval)

    print(
        f"轮询超时，{args.timeout} 秒内未等到 SUCCESS/FAILED。"
        f" 请稍后继续查询 record_id={record_id}"
    )
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("用户中断。", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"执行失败: {exc}", file=sys.stderr)
        raise SystemExit(1)