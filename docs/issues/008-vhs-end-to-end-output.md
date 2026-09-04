# [AFK] 完成 VHS 端到端视频输出

## 父问题

- 依赖：002-vhs-video-adapter
- 依赖：005-render-overlay-node

## 要构建什么

完成 `VHS Load Video → Analyze → Render → VHS Video Combine` 的端到端工作流，并验证原音频保留。

## 验收标准

- 短视频可完整跑通。
- 输出分辨率、帧率、帧数正确。
- 原音频在最终视频中保留。
- 失败时删除不完整输出并返回可读错误。
