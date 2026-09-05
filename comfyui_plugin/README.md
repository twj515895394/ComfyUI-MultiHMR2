# ComfyUI-MultiHMR2

将 Multi-HMR2 的多人 3D 人体检测、网格恢复和跨帧跟踪能力接入 ComfyUI，重点面向视频工作流。

## 1. 安装位置

推荐直接把本仓库克隆到 ComfyUI 的自定义节点目录：

```powershell
cd X:\ComfyUI-aki-v2\ComfyUI\custom_nodes
git clone git@github.com:twj515895394/ComfyUI-MultiHMR2.git
```

安装完成后的目录应为：

```text
X:\ComfyUI-aki-v2\ComfyUI\custom_nodes\ComfyUI-MultiHMR2\__init__.py
X:\ComfyUI-aki-v2\ComfyUI\custom_nodes\ComfyUI-MultiHMR2\nodes.py
X:\ComfyUI-aki-v2\ComfyUI\custom_nodes\ComfyUI-MultiHMR2\multihmr2\
X:\ComfyUI-aki-v2\ComfyUI\custom_nodes\ComfyUI-MultiHMR2\third_party\dinov3\
```

如果使用本仓库的开发目录，也可以将 `comfyui_plugin` 目录中的插件文件复制到上述目录；发布安装时推荐直接克隆仓库根目录。

## 2. 模型下载与存放

### 方式 A：手动下载（推荐）

官方 Multi-HMR2 checkpoint 下载地址：

<https://download.europe.naverlabs.com/ComputerVision/MultiHMR/multihmr2.pt>

下载后必须使用文件名 `multihmr2.pt`，放到：

```text
X:\ComfyUI-aki-v2\ComfyUI\models\multihmr2\multihmr2.pt
```

### 方式 B：命令行下载

```powershell
$modelDir = 'X:\ComfyUI-aki-v2\ComfyUI\models\multihmr2'
New-Item -ItemType Directory -Force -Path $modelDir | Out-Null
Invoke-WebRequest `
  -Uri 'https://download.europe.naverlabs.com/ComputerVision/MultiHMR/multihmr2.pt' `
  -OutFile (Join-Path $modelDir 'multihmr2.pt')
```

插件不会把 checkpoint 放在 `custom_nodes` 目录中，也不会自动把模型下载到其他目录。

插件已经内置 DINOv3 源码到 `third_party/dinov3`，正常运行不需要每次从 GitHub 下载 DINOv3。第一次安装后请重启 ComfyUI。

## 3. 依赖环境

以下命令必须使用 ComfyUI 自带 Python：

```powershell
$py = 'X:\ComfyUI-aki-v2\python\python.exe'
& $py -m pip install anny==0.6.0 roma warp-lang
& $py -m pip install pyrender trimesh matplotlib scipy
```

此外还需要：

- VHS（Video Helper Suite）：加载视频、输出视频和传递视频元数据；
- FFmpeg：VHS 和视频处理流程需要；
- CUDA 驱动与可用的 PyTorch CUDA 环境（推荐，但 CPU 也可运行，只是速度较慢）；
- `ComfyUI-RMBG`：使用 `transparent` 高质量 alpha，或使用 `original` 的原人物轮廓清除时需要。

检查 Python 和 CUDA：

```powershell
& $py -c "import sys, torch; print(sys.version); print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

## 4. 推荐视频工作流

```text
VHS Load Video
    ├─ images ───────────────┐
    ├─ frame_count ──────────┤
    ├─ audio ────────────────┤
    └─ video_info ───────────┤
                             ▼
                    MultiHMR2 Video Analyze
                             │ analysis
                             ▼
                    MultiHMR2 Video Render
                             │ images + frame_rate
                             ▼
                    VHS Video Combine
```

### MultiHMR2 Video Analyze

输入视频帧并生成每帧的人体 3D 预测、相机参数和 Track ID。分析结果会缓存到：

```text
X:\ComfyUI-aki-v2\ComfyUI\temp\multihmr2\
```

主要参数：

| 参数 | 选项/默认值 | 说明 |
|---|---|---|
| `conf_thresh` | `0.40` | 人物检测置信度；越高误检越少，但远处人物可能漏检 |
| `dist_thresh_nms` | `0.25` | 3D 骨盆距离 NMS 阈值，单位米 |
| `lowres` | `false` | 使用低分辨率人体模型，速度更快、细节更少 |
| `compile_model` | `false` | 对重复视频推理可能更快，首次运行会编译 |
| `segment_size` | `120` | 长视频分段处理帧数；分段之间共享 tracker |
| `model_size` | `384 / 512 / 640 / 768`，默认 `768` | 输入图像最大边；768 质量最好，较小值更快 |
| `track_hold_frames` | `0 / 2 / 4 / 6 / 8`，默认 `4` | 对短时漏检进行姿态插值，减少白模闪烁 |
| `inference_batch_size` | `1 / 2 / 4 / 8`，默认 `4` | 一次送入 GPU 的视频帧数；显存充足时可设为 `8`，通常比逐帧推理更快 |

### MultiHMR2 Video Render

读取 Analyze 输出并生成渲染后的 `IMAGE` 帧序列和 `frame_rate`。

主要参数：

| 参数 | 选项/默认值 | 说明 |
|---|---|---|
| `background` | `original` | 保留原视频背景 |
| `background` | `green_screen` | 输出纯绿色背景 |
| `background` | `transparent` | 输出 RGBA 透明背景 |
| `mesh_color` | `track_color` | 按 Track ID 使用稳定的多人颜色 |
| `mesh_color` | `white` | 统一不透明白模 |
| `show_mesh` | `true` | 显示 3D 网格 |
| `show_skeleton` | `true` | 显示骨骼线和关键点 |
| `show_track_id` | `true` | 显示 Track ID 文字 |
| `mesh_opacity` | `1.0` | 白模/彩色网格不透明度；1.0 表示完全不透明 |
| `track_id` | `-1` | -1 渲染全部人物；非负数只渲染指定 Track ID |
| `keep_model_loaded` | `false` | 输出后释放模型和 CUDA 缓存；连续多次运行可设为 true |

多人默认颜色按原项目的 `demo_color[track_id]` 分配。当前内置 1008 个稳定颜色：前 8 个为人工指定颜色，其余为固定随机颜色。Track ID 超过颜色表范围后循环使用。

### 原视频人物轮廓清除

`remove_source_person` 默认为 `true`。当 `background=original` 且当前帧检测到白模时，插件会调用已安装的 ComfyUI-RMBG 人像分割，先移除原视频人物像素并修复背景，再合成 Multi-HMR2 白模，从而避免原人物轮廓残留在白模外。插件优先使用 `models\\RMBG\\BiRefNet\\BiRefNet-portrait.safetensors`；如果该模型未缓存，会自动复用已存在的 `models\\RMBG\\RMBG-2.0\\model.safetensors`，不会强制联网下载。

如果没有安装 ComfyUI-RMBG，日志会提示并保留原来的合成行为。若不接受背景修复可能产生的纹理变化，建议使用 `green_screen` 或 `transparent` 背景模式。

`Lucida.safetensors` 也是背景移除模型，但它重点优化透明物体、伪装物体、文字 Logo、发光特效和插画，不作为本插件的真人视频分割模型；MultiHMR2 优先使用 `BiRefNet-portrait`。

## 5. Windows OpenGL 说明

Multi-HMR2 原生网格渲染使用 `pyrender.OffscreenRenderer`。

在 Windows 上使用系统原生 WGL，不需要安装 EGL。不要在 Windows 的全局启动环境中设置：

```text
PYOPENGL_PLATFORM=egl
```

EGL 主要用于 Linux 无窗口 GPU 渲染。Windows 上如果其他节点提前把 PyOpenGL 锁定为 `EGLPlatform`，MultiHMR2 插件会尝试重新绑定为 `Win32Platform`。

如果日志显示原生上下文仍不可用，插件会使用软件三角面回退，但原生 `pyrender` 的质量更好。

## 6. 内存与缓存

- 分析缓存位于 `ComfyUI\temp\multihmr2`，用于避免重复分析同一视频配置；
- 模型默认在 Render 输出后移回 CPU，并清理 CUDA 缓存；
- 如果需要连续运行多个 Render 节点，把 `keep_model_loaded` 设为 `true`；
- 第一次 Anny 初始化可能需要较长时间，后续会读取用户缓存；
- 清理 `temp\multihmr2` 只会删除分析缓存，不会删除模型文件。

## 7. 常见问题

### 找不到模型

确认文件存在且名称完全正确：

```text
X:\ComfyUI-aki-v2\ComfyUI\models\multihmr2\multihmr2.pt
```

### 每次下载 DINOv3

确认插件目录存在：

```text
X:\ComfyUI-aki-v2\ComfyUI\custom_nodes\ComfyUI-MultiHMR2\third_party\dinov3\hubconf.py
```

如果不存在，说明插件安装不完整；重新克隆或复制完整仓库，不要只复制 `nodes.py`。

### FFmpeg 报错

确认 `ffmpeg.exe` 已加入 PATH，或使用 VHS 自带的 FFmpeg 配置。先用 VHS 单独加载/合并视频验证视频链路。

### 白模闪烁

将 Analyze 的 `track_hold_frames` 设置为 `4`；如果仍有短时闪烁，可尝试 `6` 或 `8`。长时间消失通常代表模型确实没有检测到该人物，不建议无限制复制上一帧。

### 显存占用高

保持 Render 的 `keep_model_loaded=false`，并等待节点执行完成。若仍有其他节点占用显存，需要分别释放或重启 ComfyUI。

## 8. 上游项目与许可

本插件基于 [NAVER Multi-HMR 2](https://github.com/naver/multi-hmr2) 集成。仓库中保留上游的 `LICENSE.txt`、`NOTICE.txt` 及 DINOv3 相关许可文件；使用和再分发时请同时遵守各上游项目的许可证。

上游论文与项目说明请参阅仓库根目录 README。
