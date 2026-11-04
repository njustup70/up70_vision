# .devcontainer/pkgs —— 本地安装包投放处（内容不进版本库）

## 这里放什么

**海康 MVS SDK 的 deb 安装包**（`MVS-<版本>_x86_64_<日期>.deb`，约 184 MB）。

主 `Dockerfile` 在 `WITH_MVS=1` 时通过构建时只读挂载安装 SDK，deb 不写入镜像层——因为 MVS 是 **deb 包**，不是 pip 包，
写不进 `pyproject.toml` 的 `dependencies`，只能走镜像层。

## 怎么拿到

到海康机器人官网「服务支持 → 下载 → 机器视觉工业相机客户端 MVS（Linux x86_64）」
下载（需注册登录），解压 zip 后把 **`.deb` 那个**（不是 `.tar.gz`）放到本目录：

```
.devcontainer/pkgs/MVS-5.1.0_x86_64_20260909.deb
```

## 为什么不在版本库里

184 MB 远超「单文件超过 10 MB 就不进版本库」的约定（见 `knowledge/README.md`）。
所以 `.gitignore` 排除了本目录下的 `*.deb`，每个开发者各自下载一份放进来。

## 构建要求

默认 `WITH_MVS=0`，不安装 MVS，本目录没有 deb 也能构建 Gemini 环境。
需要海康时，本目录必须恰好有一个 `.deb`，再运行：

```bash
WITH_MVS=1 docker compose -f .devcontainer/docker-compose.yml build
```

启用后缺少或存在多个安装包均报错，安装失败不会跳过。两台相机使用同一 `vision` 服务。
SDK 会增大镜像；旧镜像的体积数据不代表本轮结果。容器使用 `/opt/MVS/lib` 和
`/opt/MVS/Samples`；宿主 udev 权限、USB 和窗口配置见[共用相机容器](../README.md)。
