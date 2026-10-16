# .devcontainer/pkgs —— 本地安装包投放处（内容不进版本库）

## 这里放什么

**海康 MVS SDK 的 deb 安装包**（`MVS-<版本>_x86_64_<日期>.deb`，约 184 MB）。

Dockerfile 会把这个目录 `COPY` 进镜像并安装它——因为 MVS 是 **deb 包**，不是 pip 包，
写不进 `pyproject.toml` 的 `dependencies`，只能走镜像层。

## 怎么拿到

到海康机器人官网「服务支持 → 下载 → 机器视觉工业相机客户端 MVS（Linux x86_64）」
下载（需注册登录），解压 zip 后把 **`.deb` 那个**（不是 `.tar.gz`）放到本目录：

```
.devcontainer/pkgs/MVS-5.1.0_x86_64_20260909.deb
```

## 为什么不在版本库里

184 MB 远超「单文件超过 10 MB 就不进版本库」的约定（见 `assets/README.md`）。
所以 `.gitignore` 排除了本目录下的 `*.deb`，每个开发者各自下载一份放进来。

## 不放会怎样

镜像照样能 build，只是**没有相机 SDK**：
`input.source=camera` 会在启动时报错并提示「MVS 未安装」。
不接相机的开发（合成图、回放、判色调参、单测）完全不受影响。

## 体积代价

装了 MVS 的镜像约 **2.36 GB**（不装时约 862 MB）——MVS 装完会把样例、GenICam CLProtocol、
图形客户端一起带进来。**不接相机的人不要把这个 deb 放进本目录**，就能省下 1.5 GB。

## 注意

容器里只用到 MVS 的**库和 Python 样例**（`/opt/MVS/lib`、`/opt/MVS/Samples`）。
**udev 权限规则必须在宿主机配**（容器里改不了），见 `README.md` 的环境章节。
