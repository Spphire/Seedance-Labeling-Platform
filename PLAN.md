# Seedance 数据标注平台独立版架构计划

## 目标

本项目是一个完全独立的本地 Web 平台，落地在 `W:\公司\seedance_labeling_platform`。平台负责从 DM3data NEDF episode 中提取 head/ego 视频，统一转为 760x570 MP4，切成 4s 到 15s 的 clips，批量 mock/Seedance 生成，人工审核，保留后 30fps 归一化，并在 episode 全部 clip accepted 后拼接完整 episode。

## 核心原则

- 依赖、配置、数据库、日志和视频产物都留在项目目录内。
- 默认 `generation_mode=mock`，首轮端到端验证不调用 Seedance API。
- 生成结果不可变，重跑创建新 generation job。
- 审核记录指向被选中的 generation job。
- final episode 是派生产物；clip 重新生成或重新审核后，final 状态会变为 `stale` 或重新拼接。

## 当前实现

- Backend: `app/backend`
  - FastAPI API 与静态服务。
  - SQLite 状态机。
  - DM3data scp 拉取、NEDF3 H.264 head 提取、clip 切片、mock/Seedance 生成、审核后处理、final 拼接。
- Frontend: `app/frontend/index.html`
  - episode 输入、预处理、已有 head MP4 导入、mock/seedance 生成、dry-run、auto-accept。
  - browser-like clip tabs。
  - 左右视频同步播放与保留/丢弃/重跑/标记问题。
- Seedance: `app/seedance/client.py`
  - `mock` backend 复制原 clip。
  - `seedance` backend 使用 `duration=ceil(clip_duration_sec)`。
  - `dry_run` 写 JSON 请求，不提交 API。

## 数据目录

- `data/episodes`
- `data/head_videos`
- `data/clips`
- `data/generated`
- `data/accepted_clips`
- `data/final_episodes`
- `db.sqlite3`
- `config/settings.json`
- `logs`

## 验证策略

- Stage 0: 项目内 `.venv` 和本地 wheel。
- Stage 1: head MP4 为 H.264、760x570、可播放。
- Stage 2: clip 切分覆盖 14.2s、32s、32.5s、46.7s、182.3s，所有 clip 在 [4s, 15s]。
- Stage 3: mock 生成不调用 Seedance API。
- Stage 4: accepted clips trim 到原 clip 时长，并重采样到 30fps。
- Stage 5: 全部 accepted 后拼接 final episode。
- Stage 6: Seedance dry-run 检查 duration 使用 ceiling。
- Stage 7: 真 API canary 需要人工显式切换 `generation_mode=seedance` 并配置 API key。
