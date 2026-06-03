# Engineering Review Log

## Scope

本轮 review 以当前工作树为准，重点检查：

- API 是否泄露敏感配置。
- 状态机是否能防止错误审核和错误重跑。
- mock / dry-run 是否能保证不误触真实 Seedance API。
- 测试是否覆盖关键约束。
- 当前阶段代码是否足够清晰，便于下一阶段并行任务和 UI 迭代。

## Findings And Actions

### P1: Public API exposed full settings

`GET /api/settings`、`GET /api/health`、`GET /api/state` 原先直接返回完整 settings。未来填入 `seedance_api_key` 后会泄露 token。

Action:

- 新增 `public_settings()`。
- 对外 API 只返回脱敏配置和 `seedance_api_key_set`。
- 保存配置仍写入完整 `config/settings.json`。
- 新增测试 `test_public_settings_do_not_expose_api_key`。

### P1: Review could accept a non-video dry-run output

Seedance dry-run 会生成 JSON payload，并创建 `succeeded` job。原审核逻辑只检查 `status='succeeded'` 和 `output_path`，理论上可能把 dry-run JSON 交给 ffmpeg。

Action:

- `accept` 现在要求 job status 为 `succeeded`，且 `output_path` 是 `.mp4`。
- 新增测试 `test_accept_rejects_dry_run_json_output`。

### P1: Review job/clip relationship needed explicit validation

传入 `job_id` 时，原逻辑没有显式校验 job 属于当前 clip。UI 正常不会传错，但 API 层应防误用。

Action:

- `review_clip` 校验 `generation_jobs.clip_id == clip_id`。
- 新增测试 `test_review_job_must_belong_to_clip`。

### P2: UUID parsing lived in service layer

schema 需要 UUID 校验时若直接依赖 services 会形成不理想的层级关系。

Action:

- 新增 `app/backend/ids.py`。
- `schema.py` 和 `services.py` 都依赖 `ids.py`。

### P2: Remote fetch subprocess decoding was platform fragile

Windows 默认 GBK 可能无法解码 scp/ssh 输出。

Action:

- `fetch_episode` 的 subprocess 输出统一 `encoding='utf-8', errors='replace'`。

## Current Known Limits

- 任务执行当前仍是 API 请求内同步执行；mock 阶段可用，真 Seedance 批量生成前建议升级为后台 job worker。
- 当前 UI 是单文件无构建前端，适合 v0/v1 验证；复杂审核协作阶段建议拆分组件并引入更严格状态管理。
- `settings` 写入没有权限控制；本地单机阶段可接受，团队共享部署前需要认证。
- 数据库 schema 没有 migration 工具；下一阶段建议引入轻量 migration。
- 后台常驻启动在当前 shell 环境中不稳定，但 `run_server.bat` / 前台 uvicorn 启动已验证可用。

## Review Verdict

当前实现已经适合继续做 mock 数据生产、人工审核体验迭代和 Seedance canary。下一阶段重点不应再堆功能，而应把长任务执行、可观测性、配置安全和测试矩阵补牢。
