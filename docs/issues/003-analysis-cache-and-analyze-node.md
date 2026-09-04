# [AFK] 实现视频分析缓存与 MultiHMR2 Video Analyze 节点

## 父问题

- 依赖：001-runtime-model-loader
- 依赖：002-vhs-video-adapter

## 要构建什么

实现逐帧 Multi-HMR 2 推理、跨帧 Track ID、人体参数保存和可复用缓存，并暴露 Analyze 节点。

## 验收标准

- 默认分析全部检测到的人物。
- 输出包含 Track ID、置信度、3D 姿态、形状参数、相机参数和必要的网格/投影数据。
- 同一输入和分析参数能够命中缓存。
- 只修改渲染参数时不会重新分析。
- 缓存中断不会产生可被误认为成功的半成品。
