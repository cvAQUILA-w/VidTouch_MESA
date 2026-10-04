# VidTouch MLLM Benchmark 实验与结果提交规范

## 1. 目的与适用范围

本文档用于统一 VidTouch 上通用多模态大模型（MLLM）和 MLLM-Fabric
微调模型的实验设置、数据划分、标签处理、输出解析与结果计算方式。

当前计划比较：

- 5 个通用 MLLM：作为 RGB-only 通用模型基线。
- 1 个使用 MLLM-Fabric 方法微调的 Llama：作为任务适配基线。
- VidTouch 终版方法：作为 RGB + tactile 的完整方法。

不同输入模态的模型必须明确标注，RGB-only 模型不能被描述为与
RGB + tactile 方法具有完全相同的输入条件。没有输出跨模态嵌入的
MLLM，其 retrieval 指标应填写 `N/A`，不能填写 0。

旧版 `raw_data/*.json` 和 `result.txt` 仅作为待校正的初步结果，不作为
正式论文成绩。旧结果使用了过时标签，并存在子串解析错误。

## 2. 唯一数据源与冻结划分

### 2.1 Source of truth

实验必须使用以下文件：

- 标签文件：[`label.txt`](label.txt)
- 冻结划分：[`fabric_common_v2.json`](fabric_common_v2.json)
- 划分说明：[`../../splits/README.md`](../../splits/README.md)

关键校验值：

| 项目 | SHA256 |
|---|---|
| `fabric_common_v2.json` | `4a466d92bfaf65812b835eb3f4c28787df4848eb4f0d7ba331a1f9a7267fe45d` |
| Fabric assignment | `c96c6c8af21b52e6a180361d0143f292baddc4892913848ecfa257beb4a29019` |
| Data manifest | `66a00fa35f3a6d2c3cc014e6f9114a528ab127552b650df51657cdd36c836602` |
| 当前 `label.txt` | `bd263e1e73118e4cafff754d5a123b6e78549ae3e1aa4a0081d14ca681b303d9` |

实验开始前必须运行：

```bash
cd VidTouch_MESA
python tools/validate_benchmark_split.py \
  --data-root /path/to/VidTouch \
  --split splits/fabric_common_v2.json
```

只有输出 `"valid": true` 且三个哈希一致时才能开始正式实验。

### 2.2 固定 Train/Val/Test

`fabric_common_v2` 是唯一正式划分：

| Partition | Fabric ID 数量 |
|---|---:|
| Train | 100 |
| Validation | 22 |
| Test | 22 |

三个集合以 Fabric ID 为单位完全互斥。训练 seed 不得重新生成划分。

必须遵守：

1. Train 仅用于模型训练。
2. Validation 用于 prompt 设计、超参数调整、checkpoint 选择和 feature threshold 校准。
3. Test 只用于方法、prompt、解析器和 checkpoint 全部冻结后的最终评测。
4. 不得根据 Test 结果修改 prompt、标签映射、视图聚合方式或选择模型版本。
5. 不得从多个 Test epoch 中挑最高结果。

## 3. 冻结标签空间

标签拼写和顺序以 split JSON 的 `allowed_labels` 为准。

### 3.1 Weave：10 类

```text
double-layer, herringbone, jacquard, plain, plain-knit,
plain-varied, rib, terry, twill, warp-knitted
```

### 3.2 Material：13 类

```text
cotton, cottonpolyamide, cottonpolyester, cottonpolyesterspandex,
linencotton, plastic, polyamide, polyester, polyestercotton,
polyesterpolyamide, polyesterspandex, polyesterviscose, wool
```

材料名称中的成分顺序有语义，不能交换。例如
`cottonpolyester` 和 `polyestercotton` 是不同标签。

### 3.3 Usage：11 类

```text
coat, dress, hoodie, knitwear, light-coat, lining,
shirt, skirt, suit, t-shirt, top
```

### 3.4 Features：14 类

```text
breathable, durable, elastic, firm, glossy, lightweight, rough,
sheer, smooth, soft, textured, thick, warm, wrinkle-resistant
```

### 3.5 标签 mask 与有效样本数

每个 Fabric ID 可以选择性参与不同属性的训练和评测：

- 单标签任务：canonical ground truth 位于该任务的冻结词表时才有效。
- Features：将 ground truth 与 14 类词表取交集；交集非空时才有效。
- 被过滤的稀有标签不进入输出类别空间，模型不能预测这些稀有标签。
- mask 只决定 loss 和 metric 是否计算，不得删除整个 Fabric ID。
- 推理时应对所有 Fabric ID 生成四项预测，评测器最后统一应用 mask。
- 不得因模型输出无效、解析失败或预测错误而把样本从分母中删除。

正确的有效 Fabric 数量如下：

| Partition | Weave | Material | Usage | Features |
|---|---:|---:|---:|---:|
| Train | 75 | 56 | 65 | 75 |
| Validation | 18 | 19 | 17 | 18 |
| Test | 17 | 16 | 17 | 18 |

如果任一模型的分母与上表不同，应停止汇总并检查标签版本、canonicalization
或评测代码。旧 benchmark 中 Material 的 `18/14` 分母不是正式口径。

## 4. 输入协议

### 4.1 公平的 RGB 输入

所有 MLLM 必须使用同一份冻结的 RGB 输入 manifest，并保存每个 Fabric ID
实际使用的相对图片路径。不能让不同模型自行挑选不同图片。

推荐将“所有可用 RGB 视图独立推理后，在 Fabric ID 层聚合”作为主协议：

1. 每张 RGB 视图分别执行相同的四项任务。
2. 单标签任务使用多数投票。
3. 单标签票数相同时，使用该 Fabric ID 按路径排序后的第一张主视图预测。
4. Features 对每个标签使用严格多数投票；恰好平票时跟随主视图。
5. 同时保存 per-view 预测和 Fabric-level 聚合预测。

该协议兼容只支持单图输入的模型。支持原生多图输入的模型可以另报
multi-image 结果，但不能与独立视图聚合结果混在同一列。

如果组员决定保留当前 single-view 实验，则必须额外提交冻结的图片 manifest，
并明确写成 `single-view RGB-only`。不能只保存 Fabric ID 而不保存图片路径。

### 4.2 禁止的数据泄漏

- 不得把商家提供的名称、材料、用途、特征文字作为推理输入。
- 不得把包含标签的文件名、目录名、表格或 OCR 文本提供给模型。
- MLLM-Fabric 的训练样本只能来自 Train Fabric ID。
- instruction tuning、hard example mining、类别重采样和数据增强均只能使用 Train。
- prompt、候选标签顺序和解析规则只能用 Train/Validation 开发。
- Test ground truth 只能在预测文件写盘并锁定后由评测器读取。

## 5. Prompt 与生成设置

### 5.1 推荐任务形式

四个属性分别询问，不使用一段自由文本同时回答四项任务。候选标签顺序按
第 3 节固定，不因模型或 partition 改变。

单标签任务必须要求严格 JSON：

```json
{"label": "plain"}
```

Features 必须要求严格 JSON 数组：

```json
{"labels": ["soft", "warm"]}
```

Features 是集合，输出顺序不影响结果；重复标签必须去重。

### 5.2 解码设置

通用 MLLM 的正式结果默认使用确定性解码：

```text
do_sample = false
temperature = 0
num_beams = 1
```

同时记录：

- 完整模型名称和参数规模
- Hugging Face revision、Git commit 或 checkpoint 哈希
- processor/tokenizer revision
- dtype 与量化方式
- 输入分辨率和图像预处理
- `max_new_tokens`
- system prompt 和 user prompt 原文
- 随机 seed
- Transformers、PyTorch、CUDA 版本
- GPU 型号

若某模型必须采样，应至少运行 seeds `7, 42, 123`，并报告均值和样本标准差。

## 6. 输出解析规则

这是正式结果中最重要的实现约束之一。

### 6.1 允许的处理

- 解析 JSON。
- 去除首尾空白。
- 去除包裹整个 JSON 的 Markdown code fence。
- 对标签进行大小写归一化后，与冻结词表做完整字符串匹配。

### 6.2 严格禁止

- 禁止 `if label in response` 形式的子串匹配。
- 禁止把 `cottonpolyamide` 解析成 `cotton`。
- 禁止把 `plain-knit` 解析成 `plain`。
- 禁止把 `t-shirt` 解析成 `shirt`。
- 禁止从解释文本中提取所有出现过的 feature。
- 禁止把 `not elastic`、`not glossy` 中的标签记为正预测。
- 禁止 fuzzy matching、编辑距离猜测或根据 ground truth 修正输出。

### 6.3 无效输出

可以预先规定一次“仅修复格式”的 retry，但必须对所有模型使用相同策略，且
retry prompt 不得包含 ground truth。

一次 retry 后仍无效时：

- 单标签任务：该有效样本计为错误，不能从分母删除。
- Features：预测集合记为空集。
- 单独统计 `invalid_output_count` 和 `invalid_output_rate`。
- 保存第一次和 retry 的完整原始响应。

## 7. 指标定义

所有计算先使用 `[0, 1]` 浮点数，最终表格乘 100 并保留两位小数。

正式评分必须使用仓库内唯一参考实现：

```text
benchmark/evaluate_benchmark.py
```

组员的推理程序只负责生成符合第 9.2 节格式的预测 JSONL，不能自行生成论文
最终分数。参考 evaluator 会从当前 `label.txt` 读取 ground truth，不信任预测
文件携带的标签；它还会检查 Fabric ID、有效分母、完整标签匹配和无效输出。

### 7.1 单标签普通准确率

对属性 \(a \in \{W, M, U\}\)：

\[
\mathrm{Acc}_a =
\frac{\sum_{i \in V_a}\mathbf{1}[\hat y_i=y_i]}{|V_a|}
\]

其中 \(V_a\) 是该 partition 中该属性有效的 Fabric ID 集合。

### 7.2 单标签 balanced accuracy

\[
\mathrm{BalAcc}_a =
\frac{1}{C_a}\sum_{c=1}^{C_a}
\frac{TP_c}{TP_c+FN_c}
\]

即先计算每个标签的 recall，再对标签等权平均。冻结 split 保证每个保留标签
都出现在 Train、Validation 和 Test 中。

### 7.3 单标签 macro-F1

\[
F1_c = \frac{2TP_c}{2TP_c+FP_c+FN_c},
\qquad
\mathrm{MacroF1}_a = \frac{1}{C_a}\sum_{c=1}^{C_a}F1_c
\]

Weave、Material 和 Usage 的 macro-F1 是补充指标，虽然不参与 Main。

### 7.4 Features micro-F1

在所有有效 Fabric 和 14 个 feature 标签上累计 TP、FP、FN：

\[
\mathrm{FeatureMicroF1} =
\frac{2\sum_c TP_c}
{2\sum_c TP_c+\sum_c FP_c+\sum_c FN_c}
\]

### 7.5 Features macro-F1

\[
\mathrm{FeatureMacroF1} =
\frac{1}{14}\sum_{c=1}^{14}
\frac{2TP_c}{2TP_c+FP_c+FN_c}
\]

旧结果中的“逐样本 F1 再平均”只能作为 `Example-F1` 诊断项，不能替代
Feature Macro-F1 或 Feature Micro-F1，也不能进入 Main。

### 7.6 Macro Main

论文的主要长尾指标：

\[
\mathrm{MacroMain} =
\frac{
\mathrm{BalAcc}_{W}+
\mathrm{BalAcc}_{M}+
\mathrm{BalAcc}_{U}+
\mathrm{FeatureMacroF1}}
{4}
\]

### 7.7 Legacy Main

频率敏感的补充指标：

\[
\mathrm{LegacyMain} =
\frac{
\mathrm{Acc}_{W}+
\mathrm{Acc}_{M}+
\mathrm{Acc}_{U}+
\mathrm{FeatureMicroF1}}
{4}
\]

不得再使用只平均 Weave、Material 和 Usage 的 `AVG3` 作为 Main。
如需保留 `AVG3`，只能放在补充材料并明确其不包含 Features。

### 7.8 Cross-modal retrieval

仅对能输出 RGB 与 tactile embedding 的方法计算：

- I2T R@K：给定 RGB Fabric query，在 tactile candidates 的前 K 名中是否出现同一 Fabric ID。
- T2I R@K：给定 tactile Fabric query，在 RGB candidates 的前 K 名中是否出现同一 Fabric ID。
- Mean R@K：I2T 与 T2I 的算术平均。
- 正式报告 K = 1 和 K = 5。
- 同一 Fabric 的重复观测必须先用冻结规则聚合，再进行 Fabric-level 排名。

## 8. 模型选择、threshold 与 seeds

### 8.1 通用 MLLM

- 模型无训练 checkpoint 时，prompt 和解析器在 Validation 冻结。
- 确定性生成可报告单次结果，但必须记录 seed 和完整环境。
- 不得根据 Test 上哪个模型更高而只报告该模型。
- 五个通用 MLLM 应全部保留并全部报告。

### 8.2 MLLM-Fabric 微调模型

正式 seeds 固定为：

```text
7, 42, 123
```

每个 seed 必须保留：

- `best_macro`：Validation Macro Main 最高的 checkpoint。
- `best_legacy`：Validation Legacy Main 最高的 checkpoint。
- `last`：最后一个 epoch 的 checkpoint。
- 完整逐 epoch 训练日志。

论文主结果使用 `best_macro`。从同一个 `best_macro` checkpoint 同时计算并报告
Macro Main 和 Legacy Main。`best_legacy` 可以作为预先声明的次要结果单列，
但不得在 Test 上比较两者后再挑高分。

### 8.3 Feature threshold

- 生成式 MLLM 直接输出 feature 集合，不做 Test threshold 搜索。
- 分类器式模型只能在 Validation 搜索 threshold。
- 默认搜索区间可设为 `0.10:0.05:0.90`。
- 先最大化 Validation feature macro-F1，平分时再比较 micro-F1。
- 每个 seed 的 threshold 单独保存，并原样用于 Test。

### 8.4 多 seed 汇总

三个 seed 的均值：

\[
\bar x = \frac{1}{S}\sum_{s=1}^{S}x_s
\]

样本标准差：

\[
s_x =
\sqrt{\frac{1}{S-1}\sum_{s=1}^{S}(x_s-\bar x)^2}
\]

其中 \(S=3\)。表格使用 `mean +/- sample std`，不能用总体标准差。

## 9. 必须提交的结果数据

### 9.1 实验配置与复现信息

每个模型必须提交：

- `config.json`：模型版本、prompt、生成参数、seed、环境和硬件。
- `split.json` 或 split 文件哈希。
- `input_manifest.json`：每个 Fabric ID 的实际图片路径及主视图。
- `prompt.txt` 或完整 prompt 模板。
- 推理和评分代码。
- `requirements.txt`、conda environment 或容器信息。

MLLM-Fabric 还必须提交：

- 训练配置和训练命令。
- 每个 seed 的逐 epoch 日志。
- `best_macro`、`best_legacy`、`last` checkpoint。
- 每个 checkpoint 对应的 Validation epoch、分数和 threshold。

### 9.2 原始预测

Val 和 Test 都要保存 JSONL；每个 Fabric ID 至少包含：

```json
{
  "protocol": "fabric_common_v2",
  "split_sha256": "4a466d92...",
  "partition": "test",
  "model": "model-name",
  "model_revision": "revision-or-hash",
  "seed": 42,
  "fabric_id": "7114",
  "image_paths": ["relative/path/image1.jpg"],
  "raw_responses": {
    "weave": "...",
    "material": "...",
    "usage": "...",
    "features": "..."
  },
  "parsed_prediction": {
    "weave": "plain-varied",
    "material": "linencotton",
    "usage": "light-coat",
    "features": ["textured"]
  },
  "valid_output": {
    "weave": true,
    "material": true,
    "usage": true,
    "features": true
  }
}
```

`parsed_prediction` 必须是已经完成 Fabric-level 聚合的最终预测。也可以将该
字段命名为 `aggregate_prediction`。

如果希望由参考 evaluator 执行第 4.1 节规定的多视图聚合，则不要提供上述两个
聚合字段，而应提供：

```json
{
  "partition": "test",
  "fabric_id": "7114",
  "view_predictions": [
    {
      "image_path": "relative/path/image1.jpg",
      "raw_responses": {
        "weave": "...",
        "material": "...",
        "usage": "...",
        "features": "..."
      },
      "parsed_prediction": {
        "weave": "plain-varied",
        "material": "linencotton",
        "usage": "light-coat",
        "features": ["textured"]
      }
    }
  ]
}
```

参考 evaluator 将按 `image_path` 排序，以第一张作为主视图，并自动执行多数
投票和平票处理。原始响应不能只保留截断后的摘要，也不能只保留
`correct: true/false`。

### 9.3 运行唯一参考 evaluator

Validation：

```bash
python benchmark/evaluate_benchmark.py \
  --predictions outputs/model_seed42_val.jsonl \
  --labels label.txt \
  --split splits/fabric_common_v2.json \
  --partition val \
  --output outputs/model_seed42_val_metrics.json
```

Test：

```bash
python benchmark/evaluate_benchmark.py \
  --predictions outputs/model_seed42_test.jsonl \
  --labels label.txt \
  --split splits/fabric_common_v2.json \
  --partition test \
  --output outputs/model_seed42_test_metrics.json
```

Evaluator 在以下情况必须终止并报错：

- 缺失或额外出现 Fabric ID。
- 同一 Fabric ID 重复。
- 记录中的 partition 错误。
- 当前标签与 split 得到的有效分母不一致。
- 某个冻结标签在当前 partition 没有 ground-truth support。

标签预测无效不会终止整次评估，而会按第 6.3 节计错并统计 invalid rate。

### 9.4 每个 partition、model、seed 的汇总

必须提供：

- 有效样本数：W/M/U/F。
- Weave：accuracy、balanced accuracy、macro-F1。
- Material：accuracy、balanced accuracy、macro-F1。
- Usage：accuracy、balanced accuracy、macro-F1。
- Features：macro-F1、micro-F1。
- Macro Main、Legacy Main。
- 每个属性的 invalid output 数量和比例。
- 每个标签的 support、precision、recall、F1。
- Weave、Material、Usage confusion matrix。
- 推理时间、峰值显存和平均每个 Fabric 的耗时。
- 若适用：I2T/T2I/Mean R@1 与 R@5。

推荐额外提供按 Test Fabric ID bootstrap 的 95% confidence interval，但它不能
替代三 seed 的均值与标准差。

## 10. 论文结果表模板

### 10.1 主识别表

| Model | Type | Input | Macro Main | Legacy Main | W Bal. | M Bal. | U Bal. | F Macro | W Acc. | M Acc. | U Acc. | F Micro |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| InternVL2.5-8B | General MLLM | RGB |  |  |  |  |  |  |  |  |  |  |
| LLaVA1.5-7B | General MLLM | RGB |  |  |  |  |  |  |  |  |  |  |
| Qwen2-VL-7B | General MLLM | RGB |  |  |  |  |  |  |  |  |  |  |
| GLM-4V-9B | General MLLM | RGB |  |  |  |  |  |  |  |  |  |  |
| Gemma-3-12B | General MLLM | RGB |  |  |  |  |  |  |  |  |  |  |
| MLLM-Fabric (Llama-3.2-11B) | Fine-tuned | RGB |  |  |  |  |  |  |  |  |  |  |
| VidTouch final method | Fine-tuned | RGB + tactile |  |  |  |  |  |  |  |  |  |  |

MLLM-Fabric 和 VidTouch 终版方法使用三个 seed，填写 `mean +/- std`。
通用 MLLM 若采用确定性推理则填写单次冻结结果。

### 10.2 Retrieval 表

| Model | Input | I2T R@1 | T2I R@1 | Mean R@1 | I2T R@5 | T2I R@5 | Mean R@5 |
|---|---|---:|---:|---:|---:|---:|---:|
| Model with paired embeddings | RGB + tactile |  |  |  |  |  |  |

不支持 tactile 或不输出可比 embedding 的 MLLM 填写 `N/A`。

## 11. 提交前验收清单

- [ ] split 文件 SHA256 正确。
- [ ] validator 输出 `valid: true`。
- [ ] Val/Test Fabric ID 数量为 22/22，且与冻结文件逐项一致。
- [ ] 有效分母为 Val `18/19/17/18`、Test `17/16/17/18`。
- [ ] 使用当前 canonical `label.txt`。
- [ ] 材料成分顺序未被交换。
- [ ] 所有模型使用同一 RGB input manifest。
- [ ] 保存了每张图片路径、原始响应和解析后预测。
- [ ] 输出使用严格 JSON。
- [ ] 没有使用子串、否定词扫描或 fuzzy matching。
- [ ] 无效输出计错而非删除。
- [ ] 最终指标由 `benchmark/evaluate_benchmark.py` 统一生成。
- [ ] AVG3 没有冒充 Main。
- [ ] 同时报告 Macro Main 和 Legacy Main。
- [ ] MLLM-Fabric 使用 seeds 7/42/123 并报告样本标准差。
- [ ] 保存 `best_macro`、`best_legacy` 和 `last`。
- [ ] Test 未参与 prompt、checkpoint、threshold 或模型版本选择。
- [ ] 通用 MLLM 与 MLLM-Fabric 微调模型在表中明确区分。
- [ ] RGB-only 与 RGB + tactile 方法在表中明确区分。
