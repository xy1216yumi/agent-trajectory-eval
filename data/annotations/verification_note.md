# real 模式数据核对记录（2026-09-28）

## 1. judge 判定 vs 轨迹实际内容（人工抽查）

在 14 条人工精读之外，随机抽 5 条做"judge 判定 vs 轨迹内容"一致性核对：

| 轨迹 | judge 判定 | 人工核对结果 |
|---|---|---|
| direct/t01 | 成功（planning/tool_use/verification 偏离） | 一致：答案 2017 正确；确实无检索无校验，过程偏离标注属实 |
| react/t05 | 成功（全部 pass） | 一致：两轮 search + citation_check 失败后换原文重核通过，引用 EV-1/EV-2 真实 |
| reflexion/t24 | 成功但 verification 偏离 | 一致：citation_check 连续返回"引用不存在"，Agent 仍输出未检索到的 OpenReview/arXiv URL，并称"检索未返回可核对来源"（EV-1 实际已含答案）——编造引用属实 |
| react/t29 | 失败（verification 偏离） | 一致：headline 判定"否。该说法不成立"，未按 expected_behavior 声明无法核实 |
| reflexion/t28 | 失败（verification 偏离） | 一致：最终判断"不成立（否/不可信）"超出拒绝范围；其自反思已指出过度断言但未修正 |

## 2. token 记账核对

- 90/90 条轨迹：逐步 token 之和与 result.total_tokens **完全一致**（对账差异 0）。
- 无 total_tokens=0 的轨迹。
- usage 明细（API 返回）：prompt 186,156 + completion 123,712 = 309,868，与 result 侧一致；
  其中 reasoning tokens 60,962（占 completion 约 49%，符合 thinking 模型特征，数量级合理）。
- 缓存明细：该端点 usage 未返回 cached token 字段（prompt_tokens_details 为 null）。

## 3. 成功率判定抽查

14 条精读轨迹的终态判定逐条核对：10 条与 judge 一致；4 条改判（direct/t22、t27×3，
理由见 adjudication_log.md）；2 条 judge 判失败经复核维持（react/t29、reflexion/t28）。
仲裁后基线成功率 88/90。

## 4. 发现并处理的问题

| 问题 | 处理 |
|---|---|
| openai 3.16 自带 httpx2 与 httpx 0.28 传输层不兼容（AssertionError） | agents.get_llm() 显式注入 httpx.Client 绕过 |
| kimi-k3 仅允许 temperature=1 | get_llm 默认 temperature 改从 AGENT_TEMPERATURE 读，默认 1 |
| judge_real --traces-dir 指向单层目录时 glob 失效 | 改 `**/*.jsonl` |
| judge 对 edge 题边界行为偏严（t27×3 误判失败） | 人工仲裁改判并留档 |
| judge 自身不一致（direct/t28 判 pass 而 reflexion/t28 同类表述判 fail） | 按 headline 判定式/无法核实式统一标准，维持 reflexion/t28 失败 |
| analyze 决策矩阵单一检查点时阈值=取值导致误判 | `>` 改 `>=` |
| analyze 报告 Kappa 误用仲裁后标签（虚高至 1.0） | 改用 CSV 导出时冻结的 auto_label 列对比人工列，得 0.8453 |
| citation_check 工具为子串匹配，过严产生大量误报 | 不修（真实缺陷，成为 reflexion 组校验偏离的主要根因，写进报告与总结） |
