# AI Driven Spatial Pathologist

`AI Driven Spatial Pathologist` 是一个独立于 `HistoSeg` 主仓库的 SciLifeLab Serve 部署包装层。

它的目标是：

- 让其他用户通过网页上传 Xenium 数据或关键输入文件
- 调用 `HistoSeg` 的 Pattern1 isoline 分析
- 下载 `params.json`、预览图和 contour `.npy` 文件

这个目录可以单独维护，不需要并入 `HistoSeg` 仓库。

## 现在这版 app 支持什么

当前包装的是 `HistoSeg` 仓库里最稳定的主流程：

- 输入：
  - `cells.parquet`
  - `clusters.csv`
  - 推荐提供 `tissue_boundary.csv`
- 输出：
  - `params.json`
  - `pattern1_isoline_<level>_<i>.npy`
  - `pattern1_isoline_<level>.png`
  - 一个额外打包好的 `histoseg_outputs.zip`

用户可以用两种方式上传：

1. 直接上传一个 Xenium 输出目录压缩包 `.zip`
2. 分别上传 `cells.parquet`、`clusters.csv`、`tissue_boundary.csv`

如果 zip 里包含下面这些路径，app 会自动识别：

- `cells.parquet`
- `tissue_boundary.csv`
- `analysis/clustering/gene_expression_graphclust/clusters.csv`

## 重要限制

这版 app 不是“任意原始 Xenium 数据一上传就自动完成所有上游分析”。
它依赖 `HistoSeg` 当前已有的 Pattern1 isoline 工作流，所以本质上需要：

- 细胞表 `cells.parquet`
- 聚类结果 `clusters.csv`

如果你的 Xenium 输出 zip 里已经有 `analysis/clustering/gene_expression_graphclust/clusters.csv`，那么通常就可以直接跑。
如果没有，就需要在 app 里再补一段上游 clustering 流程，这是下一阶段工作，不在当前脚手架范围内。

## 本地运行

```bash
cd AI-Driven-Spatial-Pathologist
python -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
python main.py
```

Windows PowerShell:

```powershell
cd D:\GitHub\AI-Driven-Spatial-Pathologist
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
python .\main.py
```

默认端口是 `7860`。

## Docker 构建

在这个目录下构建镜像：

```bash
docker build --platform linux/amd64 -t <dockerhub-user>/ai-driven-spatial-pathologist:v1 .
```

本地测试：

```bash
docker run --rm -p 7860:7860 <dockerhub-user>/ai-driven-spatial-pathologist:v1
```

然后打开：

- `http://localhost:7860`

## 推送镜像

```bash
docker push <dockerhub-user>/ai-driven-spatial-pathologist:v1
```

注意：

- SciLifeLab Serve 要求镜像是公开可访问的
- 每次更新都必须使用新的 tag
- 不要重复使用同一个 tag 覆盖旧版本

## GitHub Actions 自动构建镜像

如果你不想在本地装 Docker，可以直接把这个目录放到一个公开 GitHub 仓库里。

本目录已经包含：

- `.github/workflows/docker-image-ghcr.yml`

它会在你 push 到 `main` 后自动构建镜像并发布到：

- `ghcr.io/<你的GitHub用户名>/ai-driven-spatial-pathologist:sha-<commit>`

同时也会更新：

- `ghcr.io/<你的GitHub用户名>/ai-driven-spatial-pathologist:latest`

注意：

- Serve 更新版本时不要用 `latest`
- 应该使用唯一 tag，比如 `sha-abc1234`
- GitHub Container Registry 的 package 需要设为 `Public`，否则 Serve 拉不到

## 上传到 SciLifeLab Serve

参考官方文档：

- [Gradio app hosting](https://serve.scilifelab.se/docs/application-hosting/gradio/)
- [Other app types](https://serve.scilifelab.se/docs/application-hosting/other/)
- [File management](https://serve.scilifelab.se/docs/files/)

推荐按 `Gradio app` 类型部署。

### Serve 中建议填写

- Name: `AI Driven Spatial Pathologist`
- Description: A browser-based HistoSeg app for Xenium Pattern1 isoline analysis.
- Keywords: `xenium, spatial transcriptomics, histoseg, pathology, contour`
- Permissions:
  - 开发阶段推荐 `Link`
  - 官方文档目前提示 `Private` 和 `Project` 对 Gradio 有已知 bug，尽量不要用
- Port: `7860`
- Image: `<dockerhub-user>/ai-driven-spatial-pathologist:v1`
- Mount path:
  - 推荐在项目 Storage 里配置 `/home/username/app/project-vol`
  - 这样 app 输出和临时工作目录可以写到持久卷

如果你改成 GitHub Actions + GHCR 路线，那么 Image 直接填：

- `ghcr.io/<你的GitHub用户名>/ai-driven-spatial-pathologist:sha-<commit>`

### Source code URL 不能省略

Serve 的 app 表单要求提供 `Source code URL`。

这个 URL 不一定非要是 GitHub，但必须是一个可访问的公开地址，例如：

- 独立 GitHub 仓库
- Zenodo 记录
- Figshare
- 其他公开代码归档地址

如果你不想把这些部署文件放进 `HistoSeg` 主仓库，完全可以：

1. 保持这个目录独立
2. 把它单独发布到另一个公开位置
3. 在 Serve 表单中把那个地址填成 `Source code URL`

最方便的做法通常是：

1. 建一个公开 GitHub 仓库 `AI-Driven-Spatial-Pathologist`
2. 把这个目录内容推上去
3. 让 GitHub Actions 自动生成公开 GHCR 镜像
4. 把这个 GitHub repo URL 作为 `Source code URL`

## 许可证提醒

`HistoSeg` 当前仓库声明的是 `PolyForm Noncommercial 1.0.0`。

这意味着：

- 学术/非商业用途通常没问题
- 商业用途需要额外许可

如果你的 Serve app 面向科研协作，通常是合理的，但正式公开前仍建议你再确认一次使用场景是否符合许可证要求。

## 下一步建议

如果你希望真正做到“只上传原始 Xenium 数据，完全不用再准备 clusters.csv”，下一步可以继续扩展这套 app：

1. 在上传后自动解包 Xenium outs
2. 自动检查是否已有 `analysis/clustering/gene_expression_graphclust/clusters.csv`
3. 如果没有，就在 app 内部补跑上游 clustering
4. 再把聚类结果送进 `HistoSeg` 的 isoline 流程

当前这套脚手架已经足够支持第一版上线和共享测试。
