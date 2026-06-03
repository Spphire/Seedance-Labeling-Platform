# Future Plan And Test Settings

## Phase 1: Stabilize Local Mock Workflow

Goal:

让本地 mock 流程成为每日可重复跑的基线，保证不会误调用 Seedance API。

Deliverables:

- 固定测试数据清单：三个已知 UUID。
- `generation_mode=mock` 默认锁定。
- `dry_run` 请求 JSON 保存到 `data/generated/.../*.json`，不进入审核 accept。
- 每次改动后运行 unit tests 和 smoke tests。

Test settings:

```json
{
  "generation_mode": "mock",
  "public_base_url": "http://106.14.2.243:18080",
  "mock_concurrency": 8,
  "seedance_concurrency": 3,
  "dm3_host": "DM3data",
  "dm3_nedf_root": "/mnt/nm_data/data/nedf"
}
```

Gate:

- `python -m unittest discover -s tests -p "test_*.py"` 通过。
- 三个 UUID 的 clips 均在 `[4, 15]` 秒。
- 所有 generated jobs 为 `mode=mock,status=succeeded`。
- 所有 accepted clips 为 H.264、760x570、30fps。
- final episodes 为 H.264、760x570、30fps。

## Phase 2: Job Worker And Observability

Goal:

把长任务从 HTTP 请求中拆出去，支持可恢复、可暂停、可并发限流的 worker。

Deliverables:

- `jobs` 服务层：preprocess、generation、postprocess、stitch 统一 job runner。
- job event log：每个 job 的 started/succeeded/failed/retried。
- UI 显示进度、错误详情、重试按钮。
- Seedance API 调用增加 rate limit 和 retry backoff。

Test settings:

```json
{
  "mock_concurrency": 8,
  "seedance_concurrency": 1,
  "job_poll_interval_sec": 2,
  "max_retry_count": 2
}
```

Gate:

- 杀掉 worker 后重启，pending/running job 能恢复到可继续状态。
- 失败 job 不影响其他 episode。
- 重跑生成新 job，不覆盖旧 output。

## Phase 3: Seedance Dry-Run And Canary

Goal:

在不消耗 token 的情况下验证请求 payload，再只用一个短 clip 做真 API canary。

Deliverables:

- dry-run payload schema snapshot test。
- canary UI 开关，需要显式确认。
- canary 只允许选择一个短 clip。

Test settings:

```json
{
  "generation_mode": "seedance",
  "seedance_concurrency": 1,
  "seedance_resolution": "480p",
  "seedance_ratio": "4:3",
  "canary_max_clip_sec": 6
}
```

Gate:

- dry-run 中 `duration == ceil(clip_duration_sec)`。
- `seedance_api_key` 不出现在任何 public API response。
- canary 结果下载后能进入普通 review 流程。

## Phase 4: Review Workbench Upgrade

Goal:

提高人工审核效率，支持 episode 级进度、键盘操作、问题标签和批量复核。

Deliverables:

- Clip tab 支持搜索、筛选、排序。
- 同步播放支持快捷键。
- 问题标签枚举化。
- 审核历史可见。
- Episode 全部 accepted 后自动提醒 final ready。

Gate:

- 多 tab 不丢状态。
- review 操作写入 reviews 表。
- reject/flag 不触发 final stitching。

## Phase 5: Dataset Export

Goal:

把 accepted/final 数据整理成下游训练可消费的数据集。

Deliverables:

- manifest JSONL。
- episode-level metadata。
- 原 clip / generated / accepted / final 的映射关系。
- 校验脚本：文件存在、codec、fps、duration drift。

Gate:

- manifest 中每条 accepted clip 都能在磁盘找到。
- final duration 与 accepted clips 总时长误差在容忍范围内。
- 没有 dry-run JSON 被导出为 accepted video。
