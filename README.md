# Seedance Labeling Platform

一个完全独立的本地 Web 标注平台，用于从 DM3data 的 NEDF3 episode 中提取 head 视频、转为 760x570 MP4、切分 clip、批量生成、人工审核、保留后处理和最终 episode 拼接。

所有依赖、配置、数据库、日志和视频产物都保存在本项目目录内。

## Quick Start

```powershell
cd W:\公司\seedance_labeling_platform
.\setup.ps1
.\run_tests.ps1
.\run_server.ps1
```

默认监听 `0.0.0.0:18080`。在 DM3data 上部署后打开 `http://106.14.2.243:18080`；本地调试可打开 `http://127.0.0.1:18080`。

如果 PowerShell 执行策略禁止 `.ps1`，可以使用同目录下的 `setup.bat`、`run_tests.bat`、`run_server.bat`。

## DM3data 临时 HTTP 部署

当前默认面向 `106.14.2.243:18080` 临时部署：

```bash
cd /path/to/seedance_labeling_platform
bash setup.sh
bash run_tests.sh
bash run_server.sh
```

管理员开放 TCP `18080` 后，标注员访问 `http://106.14.2.243:18080/label`，管理员访问 `http://106.14.2.243:18080/admin`。管理员页查看全局进度、Seedance 用量、标注员活动，并维护多套 prompt/ref 图组合。

平台生成给 Seedance 的 clip URL 默认也是 `http://106.14.2.243:18080/clips/...`。如果后续切到 OSS 预签名 URL，只需要在 UI 的 Settings 里修改 `public_base_url`，已有 clip 记录会自动刷新。

本地调试可以覆盖监听地址和端口：

```powershell
$env:SEEDANCE_HOST="127.0.0.1"
$env:SEEDANCE_PORT="12222"
$env:SEEDANCE_RELOAD="1"
.\run_server.ps1
```

## 默认安全模式

默认 `generation_mode=mock`，生成阶段只复制原始 clip 到 `data/generated`，不会调用 Seedance API。

真实 Seedance 生成需要在本次运行模式中显式切到 `seedance`，并由管理员填写 Seedance API Key。标注页不展示 `dry-run` 按钮，普通标注员只使用“批量生成”和右侧 clip “重跑”。

## 真实 Seedance 配置

API key 不要提交到 Git。服务器部署后有两种配置方式：

- 在右上角齿轮的后台管理员设置里填写 `Seedance API Key` 并保存。它会写入服务器本机被 `.gitignore` 忽略的 `config/settings.json`。
- 或者用环境变量启动服务：`SEEDANCE_API_KEY=... bash run_server.sh`。兼容实验脚本里的 `ARK_API_KEY`，但 `SEEDANCE_API_KEY` 优先级更高。

切到真实生成前，先在标注页选择 Prompt/ref 图组合，必要时做本次临时微调，再确认运行模式为 `seedance（真实消耗额度）`。如需检查请求 payload，由负责人或技术人员通过后端 `dry_run=true` 调用或日志验证；标注页不再暴露独立 dry-run 按钮。

## 多人协作

- 打开 clip 标签页时会自动锁定它所在的 episode，持锁者才能保留、丢弃、重跑或标记问题。
- 锁是可续租 lease，页面打开时自动续租；关闭同一 episode 的最后一个标签页会释放。浏览器异常退出后锁会自动过期。
- 其他人可以只读查看被锁 episode，也可以显式“接管锁”。
- 预处理、导入 head、手动重新合成 final 这类 episode 级操作会拿 episode 锁；若该 episode 内有 clip 正在被别人审核，会被拒绝。
- 批量生成会跳过正在被人工锁定的 episode，避免覆盖别人正在审核的片段。
- 自动 final 拼接是后台派生产物，不需要人工 episode 锁；它会检查 clip 状态，过期结果不会覆盖新结果。

标注员日常操作请参考 [Seedance 标注员 SOP](docs/ANNOTATOR_SOP.md)。

## Data Layout

- `data/episodes`：从 DM3data 拉取的原始 episode。
- `data/head_videos`：完整 head MP4，统一 760x570。
- `data/clips`：切好的原始 clips，也是 `/clips` 静态服务根目录。
- `data/generated`：mock 或 Seedance 生成结果。
- `data/accepted_clips`：审核保留后 trim + 30fps 的 clips。
- `data/final_episodes`：全部 clips accepted 后拼接的完整 episode。
- `data/final_dataset`：final episode 回写后的 Seedance NEDF 数据集副本。
- `db.sqlite3`：可恢复状态机数据库。
- `config/settings.json`：DM3data、public URL、Seedance 后端等配置。

## Useful API

- `POST /api/pipeline/submit_preprocess`：导入 episode 并准备 head 视频。
- `POST /api/pipeline/import_head`：导入已有 head MP4。
- `POST /api/episodes/{uuid}/anchor_candidates`：按起点创建候选锚点。
- `POST /api/generation/rolling_run`：批量运行当前可处理的 mock 或 Seedance rolling 生成队列。
- `POST /api/review/{clip_id}`：保留、丢弃或标记问题。
- `POST /api/clips/{clip_id}/retry`：重跑当前 clip。
- `POST /api/locks/acquire`、`POST /api/locks/renew`、`POST /api/locks/release`：多人协作锁。
- `POST /api/test/auto_accept`：测试用，自动保留全部 mock 结果并触发最终拼接；标注页不展示这个入口。
