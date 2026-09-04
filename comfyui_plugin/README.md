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
- `MultiHMR2 Video Render`：读取分析缓存，输出 `IMAGE` 帧序列和 FPS，连接 VHS `Video Combine`。

安装依赖：

```powershell
X:\ComfyUI-aki-v2\python\python.exe -m pip install anny==0.6.0 roma warp-lang
X:\ComfyUI-aki-v2\python\python.exe -m pip install pyrender
```
