# amd-green-boost
An experimental AMD-style memory-assisted acceleration project for local LLM inference
# Green Boost v1.27  
# Green Boost v1.27（中英双语版）

> Make AMD + RAM behave more like expandable VRAM for large-model offload under HIP / llama.cpp.  
> 让 AMD + 系统内存在 HIP / llama.cpp 路线下，尽可能表现得像“可扩展显存”。

---

## Overview ｜ 项目简介

**Green Boost** is an AMD-oriented large-model inference enhancement project.  
Its goal is to push **llama.cpp + HIP** beyond the hard limit of physical VRAM by allowing **system RAM to participate in GPU-side model residency and offload**.

**Green Boost** 是一个面向 AMD 平台的大模型推理增强项目。  
它的目标是在 **llama.cpp + HIP** 路线下，突破物理显存的硬限制，让**系统内存参与 GPU 侧模型驻留与 offload**。

This release package is reconstructed from a real debugging and validation process.  
It is intended for publishing, documentation, validation sharing, and patch distribution.

这个整理包来源于一次真实的联调与验证过程。  
它适合用于发布说明、验证记录、补丁分享，以及后续整理为正式仓库的起点。

---

## Goal ｜ 核心目标

Run very large models on AMD hardware with **aggressive GPU offload** while avoiding immediate failure from VRAM exhaustion.

在 AMD 硬件上，以**尽可能激进的 GPU offload** 运行超大模型，同时避免因为显存不够而立刻失败。

In plain words:

> **Use RAM to support GPU-side model loading, so `ngl=999` becomes practically achievable.**

说人话就是：

> **让 RAM 给 GPU 侧模型承载兜底，让 `ngl=999` 从参数幻想变成可实际跑通。**

---

## What problem does it solve? ｜ 它解决什么问题？

On consumer AMD GPUs, especially around **24GB VRAM**, running **70B to 120B class models** usually hits several problems:

在消费级 AMD 显卡上，尤其是 **24GB VRAM** 这一档，运行 **70B 到 120B** 级模型时，通常会遇到这些问题：

- GPU offload depth is too low  
  GPU offload 层数太低
- VRAM fills up quickly and crashes  
  显存很快打满然后崩掉
- `ngl` cannot be pushed high enough  
  `ngl` 根本推不上去
- System RAM is large, but not effectively used for GPU-side residency  
  系统内存明明很大，却没有被有效用于 GPU 侧承载
- Default allocator behavior is too conservative  
  默认分配策略太保守

**Green Boost v1.27** pushes this boundary further by validating the managed/unified memory path at the allocator level.

**Green Boost v1.27** 通过在 allocator 层验证 managed/unified memory 路径，把这个边界往前推了一大截。

---

## Verified Result ｜ 已验证结果

The following has been **actually observed in runtime logs**:

以下结果已经在**真实运行日志中被验证**：

- `offloaded 37/37 layers to GPU`
- `ROCm0 model buffer size ≈ 117943.88 MiB`
- `CPU_Mapped model buffer size ≈ 586.82 MiB`
- `model loaded`
- `server is listening`

This means:

这意味着：

> **The model was almost fully pushed into the GPU-side residency path, while actual memory backing came from a combination of VRAM and system RAM.**

> **模型几乎被完整推进了 GPU 视角下的承载路径，而底层实际依赖的是 VRAM + 系统内存共同兜底。**

In short:

简而言之：

## **“RAM acting like VRAM” has been practically validated.**  
## **“RAM 当 VRAM” 这条路线，已经被实际验证跑通。**

---

## Hardware Platform ｜ 硬件平台

### Host Machine ｜ 宿主机配置

- **CPU**: AMD Ryzen 9 9950X
- **RAM**: about 192GB
- **GPU**: AMD Radeon RX 7900 XTX 24GB
- **OS**: Ubuntu Desktop

### Notes ｜ 备注

This is not a datacenter-class MI-series setup.  
It is a consumer-grade AMD GPU environment pushed very aggressively.

这不是 MI 系列那种数据中心卡环境。  
这是一个被狠狠干到极限的消费级 AMD 显卡环境。

---

## Main Models Used ｜ 主要测试模型

- **GPT-OSS 120B**  
  `Huihui-120b-Full.gguf`

- **Qwen 3.5 122B**  
  `Qwen3.5-122B-A10B-heretic.Q8_0.gguf`

- **Llama 4 Scout 17B**

- **Qwen 3.5 35B A3B**

---

## Core Idea ｜ 背后原理

### 1. Managed / Unified Memory ｜ 托管内存 / 统一内存

A key part of this validation relies on:

本次验证的关键之一是：

- `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1`

Even though the name still says “CUDA”, under HIP build this path maps into AMD-compatible managed memory behavior.

虽然名字里还带着 “CUDA”，但在 HIP 构建下，这条路径会映射到 AMD 兼容的 managed memory 行为。

This is one of the core mechanisms that allows GPU-side allocations to avoid immediate hard failure when physical VRAM is insufficient.

这正是 GPU 侧分配在物理显存不足时，不会立刻硬崩的重要机制之一。

---

### 2. HSA_XNACK ｜ HSA_XNACK

Another important switch:

另一个关键开关是：

- `HSA_XNACK=1`

This is highly relevant to page migration and fault-handling behavior on AMD platforms.  
Without it, the RAM-backed GPU residency path is much less convincing.

它和 AMD 平台上的页面迁移、缺页处理能力密切相关。  
没有它，RAM 参与 GPU 侧承载这条路通常会差很多。

---

### 3. `ngl=999` ｜ `ngl=999` 的含义

`ngl=999` does **not** literally mean the model has 999 layers.

`ngl=999` **并不是真的表示模型有 999 层**。

It means:

它的意思其实是：

> **Offload as many layers as possible.**  
> **尽可能把能 offload 的层全都 offload。**

In this project, `ngl=999` is the practical “full push” mode.

在本项目里，`ngl=999` 可以理解成一种“尽量全上”的实际模式。

---

## Why do AMD builds still show “cuda” names? ｜ 为什么 AMD 环境里还会看到 “cuda” 命名？

This is a historical naming issue in `llama.cpp` / `ggml`.

这是 `llama.cpp` / `ggml` 历史遗留的命名问题。

You may still see things like:

你仍然会看到类似这些名字：

- `ggml-cuda.cu`
- `cudaMallocManaged`
- CUDA-style file names and symbols

But under HIP build, the actual backend path is AMD/ROCm/HIP.

但在 HIP 构建下，真正跑的后端其实是 AMD/ROCm/HIP。

So:

所以：

> **CUDA-like naming does not necessarily mean NVIDIA runtime is being used.**  
> **看到 CUDA 风格命名，不等于真的在跑 NVIDIA CUDA runtime。**

It is mostly naming debt, not architectural betrayal.

这更多是命名债务，不是技术叛变。

---

## What changed in v1.27? ｜ v1.27 改了什么？

The most important change in v1.27 is:

v1.27 最重要的变化是：

## **Observation and validation were pushed down to the allocator layer.**  
## **观察与验证被下沉到了 allocator 层。**

Earlier work mostly handled:

更早的工作更多集中在：

- controller
- policy
- bridge
- counters
- KV lifecycle hooks

But v1.27 went lower, into actual GPU-side memory allocation behavior.

而 v1.27 更进一步，触碰到了真正的 GPU 侧内存分配行为。

That is why this version finally produced convincing evidence.

这也是为什么这个版本终于拿到了足够硬的验证结果。

---

## Project Layout ｜ 项目结构

```text
greenboost_v1.27/
├── README.md
├── patches/
│   └── 0001-greenboost-v1.27-allocator-managed-memory-probe.patch
├── configs/
│   └── config.yaml.example
├── docs/
│   ├── VALIDATION.md
│   ├── HARDWARE_AND_MODELS.md
│   ├── llama_swap_503_analysis.md
│   └── GITHUB_RELEASE_NOTES.md
└── scripts/
    └── check_llama_swap_timeout.sh
