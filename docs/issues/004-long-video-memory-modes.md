# [AFK] 增加低分辨率与长视频分段处理

## 父问题

- 依赖：003-analysis-cache-and-analyze-node

## 要构建什么

为 Analyze 增加显存友好的低分辨率模式和分段处理。分段结果需要正确合并，不能破坏帧顺序和 Track ID。

## 验收标准

- 可选择低分辨率 Anny 模式。
- 可配置分段长度并在处理后合并缓存。
- 分段边界处帧数、顺序和 Track ID 行为有测试。
- 显存不足时给出可操作的错误或降级提示。

## 实现状态

- 已实现按 `segment_size` 分段处理。
- 分段间复用同一 `FeatPelvisTracker`，保持 Track ID 连续。
