# [AFK] 实现背景模式与合成策略

## 父问题

- 依赖：005-render-overlay-node
- 依赖：006-background-segmentation-adapter

## 要构建什么

支持原始背景、透明背景和指定图片背景，并定义人物网格/骨骼图层与背景 mask 的合成顺序。

## 验收标准

- 原始背景是默认模式。
- 透明背景模式正确输出 alpha 语义或 ComfyUI 可用的替代表示。
- 指定图片背景支持尺寸适配和裁剪策略。
- 多人物 mask 不互相覆盖错误。
