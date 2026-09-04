# [AFK] 实现 MultiHMR2 Video Render 节点

## 父问题

- 依赖：002-vhs-video-adapter
- 依赖：003-analysis-cache-and-analyze-node

## 要构建什么

实现基于分析缓存的渲染节点，输出带人体效果的 `IMAGE` 帧序列和 FPS，不重复执行模型推理。

## 验收标准

- 可独立开关网格、骨骼、Track ID、置信度和 3D 框/地面位置。
- 颜色、透明度和 Track ID 筛选有效。
- 默认保留原始背景。
- Render 参数变化不会触发 Analyze。
- 输出可直接接 VHS Video Combine。
