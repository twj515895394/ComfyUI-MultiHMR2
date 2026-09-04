# ComfyUI-MultiHMR2

安装目录：

```text
X:\ComfyUI-aki-v2\ComfyUI\custom_nodes\ComfyUI-MultiHMR2
```

模型目录：

```text
X:\ComfyUI-aki-v2\ComfyUI\models\multihmr2\multihmr2.pt
```

当前版本节点：

- `MultiHMR2 Video Analyze`：接收 VHS 的 `IMAGE / frame_count / audio / video_info`，执行多人 3D 重建、跟踪与缓存。
- `MultiHMR2 Video Render`：读取分析缓存，输出 `IMAGE` 帧序列和 FPS，连接 VHS `Video Combine`。背景支持 `original`、`green_screen` 和 `transparent`；透明模式优先调用 ComfyUI-RMBG 的 BiRefNet-portrait，并输出 RGBA IMAGE。
- `segment_size` 默认每 120 帧切段，但切段之间复用同一个 tracker，保持 Track ID 连续。
- `track_id=-1` 表示渲染全部人物，设置为非负数可只渲染指定 Track ID。

安装依赖：

```powershell
X:\ComfyUI-aki-v2\python\python.exe -m pip install anny==0.6.0 roma warp-lang
X:\ComfyUI-aki-v2\python\python.exe -m pip install pyrender
```
