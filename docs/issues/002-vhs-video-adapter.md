# [HITL] 对齐 VHS 视频输入输出契约

## 要构建什么

实现 VHS 视频帧、FPS、音频/元数据与插件内部数据结构之间的适配器，确保 Analyze 接收 VHS 输入，Render 输出可交给 VHS Video Combine 的 `IMAGE` 帧和 FPS。

## 验收标准

- 使用当前安装的 VHS 完成加载短视频并读取帧、FPS、元数据。
- 输出帧数、分辨率、顺序和 FPS 正确。
- 音频/视频元数据能够传递给 VHS 编码节点。
- VHS 接口变化被隔离在适配器中。

## 被阻塞于

- 需要基于当前 ComfyUI/VHS 安装进行一次接口确认。
