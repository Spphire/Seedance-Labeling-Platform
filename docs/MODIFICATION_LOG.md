# Modification Log

## 2026-06-01 Review Pass

本轮目标是由专业软件工程视角复审当前 Seedance 标注平台，并把高价值优化、未来计划和测试 setting 落地。

## Code Changes

- 新增 `app/backend/ids.py`
  - 集中管理 UUID 解析和校验。
  - 避免 `schema.py` 反向依赖 `services.py`。

- 更新 `app/backend/schema.py`
  - `PreprocessRequest.uuids` 现在会校验并规范化 UUID。
  - `ImportHeadVideoRequest.uuid` 现在会校验并规范化 UUID。

- 更新 `app/backend/settings.py`
  - 新增 `public_settings()`。
  - 对外隐藏 `seedance_api_key`，只暴露 `seedance_api_key_set`。

- 更新 `app/backend/main.py`
  - `/api/settings`、`/api/health`、`/api/state` 改为返回脱敏 settings。
  - 移除未使用 import。

- 更新 `app/backend/services.py`
  - `review_clip` 校验传入 job 必须属于当前 clip。
  - `accept` 只允许接受成功的 `.mp4` generation output，防止 dry-run JSON 误进入审核后处理。

- 更新 `app/backend/nedf.py`
  - `scp` subprocess 输出使用 UTF-8 容错解码，降低 Windows 编码风险。

## Test Changes

- `test_public_settings_do_not_expose_api_key`
  - 验证 public settings 不返回 `seedance_api_key`。

- `test_accept_rejects_dry_run_json_output`
  - 验证 dry-run JSON 不能被 accept。

- `test_review_job_must_belong_to_clip`
  - 验证 job/clip 关系错误时审核会失败。

## Documentation Changes

- 新增 `docs/ENGINEERING_REVIEW.md`
  - 记录本轮 review 范围、发现、修复和剩余风险。

- 新增 `docs/FUTURE_PLAN_AND_TEST_SETTINGS.md`
  - 记录未来阶段目标、deliverables、测试 setting 和 gate。

- 新增本文件 `docs/MODIFICATION_LOG.md`
  - 记录本轮 review 后的实际修改。

## Verification

- `python -m unittest discover -s tests -p "test_*.py"`：9 tests passed。
- `python -m compileall app tests`：passed。
- 现有样本数据库状态：
  - 3 episodes final_status = ready。
  - 33 clips status = accepted。
  - 33 mock generation jobs status = succeeded。
