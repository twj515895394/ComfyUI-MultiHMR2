# [AFK] 建立 ComfyUI 运行时与 Multi-HMR2 模型加载层

## 要构建什么

在 ComfyUI 自定义节点环境中安全导入现有 Multi-HMR 2 后端，检查依赖并从固定目录加载 `multihmr2.pt`。模型会话需要复用，不能重复初始化。

## 验收标准

- ComfyUI 使用自带 Python 启动时，插件缺依赖会给出清晰提示。
- 模型固定从 `models/multihmr2/multihmr2.pt` 读取。
- 首次执行加载一次模型，后续执行复用会话。
- 不覆盖或降级 ComfyUI 已有的 Torch/CUDA。
- CPU/GPU 设备选择和显存清理行为可诊断。

## 被阻塞于

- 无。可立即开始。
