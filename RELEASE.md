# 发布流程

GitHub Releases 是官方发布源，Gitee Releases 是国内下载镜像。Gitee 不单独构建：必须同步 GitHub Release 上的同一份 ZIP、`.zip.sha256` 和 `release-manifest.json`，不得在两边分别打包或重生成发行文件。

1. 在干净的 `main` 上运行测试，并用 `python tools/build_release.py --version 1.2.0`（替换为目标版本号）构建本地发行目录。确认 `release-manifest.json` 中的 `git_commit` 对应当前提交、`git_dirty` 为 `false`，且 ZIP 的 SHA256 与 checksum 和 manifest 一致。
2. 创建并发布对应的 GitHub Release（例如 tag `v1.2.0`），上传 `release/` 中生成的 ZIP、`.zip.sha256` 和 `release-manifest.json`。
3. 从本机将这三份原文件同步到 [Gitee Releases](https://gitee.com/Gibran_Halili/NEUQ-VisionCalib/releases)。使用本机环境中的 `GITEE_TOKEN`，不要把令牌写入命令、文件或日志。已有同名附件时先核对，不一致时停止，不要静默覆盖。
4. 从 Gitee 重新下载三份附件，逐个核对文件大小和 SHA256；ZIP 的 SHA256 还必须与 `.zip.sha256` 和 manifest 相符。

GitHub Actions 的 **Mirror release to Gitee** workflow 仅保留 `workflow_dispatch`，可手动指定已发布的 GitHub Release tag 作为备用同步入口；它同样只镜像 GitHub 上的原始附件，不构建 ZIP。
