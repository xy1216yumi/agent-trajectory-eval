# 人工抽检与仲裁记录（real 模式，2026-09-28）

抽检 14 条轨迹（覆盖三组配置 × 各难度层 × edge 全型），以《偏离判据手册》(docs/02) 为唯一标准
独立精读，再与 judge_real 输出比对。

## 一致性

- 检查点标签：56 项中一致 52 项，Cohen's Kappa = 0.8453（analyze.py --kappa 计算）。
- 分歧 4 处，全部位于 edge 题的边界行为判定，仲裁如下。

## 仲裁明细（不一致处，以人工精读为准）

| 轨迹 | 检查点 | judge | 人工 | 仲裁理由 |
|---|---|---|---|---|
| direct/t27 | verification | deviated | pass | Agent 明确"两者都有出处、口径不同"，并列解释 300B(训练量)/499B(语料池) 两口径，未武断判定来源B错误；"若二选一取A"是回答用户字面追问的条件化指导。满足 report_conflict 实质与全部 key_conditions |
| react/t27 | verification | deviated | pass | 同上：明确"并非必然互相矛盾"，表格并列两口径并给出严谨建议表述，引用 EV-1/EV-2 真实存在 |
| reflexion/t27 | verification | deviated | pass | 同上：表格并列呈现，区分"训练通过量 vs 候选语料池"，符合 expected_behavior |
| react/t29 | planning | deviated | pass | 5 轮多角度检索流程完整、顺序正确，失败 purely 在校验环节（headline 判定"否"），规划本身无缺陷 |

## 成功判定修正（checkpoint 标签不变，仅终态判定）

| 轨迹 | judge | 人工 | 理由 |
|---|---|---|---|
| direct/t22 | 失败 | 成功 | gt 被引量口径默认 Google Scholar；Agent 明确"GS 口径下成立"并给出表格，结论与 gt 一致，多口径说明属于加分项 |
| direct/t27 | 失败 | 成功 | final_behavior 实为 report_conflict（见上） |
| react/t27 | 失败 | 成功 | 同上 |
| reflexion/t27 | 失败 | 成功 | 同上 |

## 复核后维持 judge 判失败的

- react/t29：headline 判定"否。该说法不成立"。对比同题 direct/reflexion 的 headline
  （"不能证实/尚未公布"），本轨迹把"无法核实"的问题给出了确定否定。维持 verification=deviated、失败。
- reflexion/t28：最终判断"该说法不成立（否/不可信）"超出"拒绝给出数字并说明"的期望行为；
  其自反思环节已指出该过度断言但最终答案未修正。维持失败。

## 仲裁后基线失败集（进入干预实验）

react/t29、reflexion/t28（共 2 条，均为 edge 题）。

---

# 追加：任务集 v2（t31–t45）失败轨迹复核（2026-09-28）

v2 新增 45 条轨迹，judge 判失败 9 条（direct 4 / react 3 / reflexion 2），逐条人工复核：

| 轨迹 | judge 判定 | 人工复核 | 理由 |
|---|---|---|---|
| direct/t32 | 失败（先验答 2012） | 维持 | 未检索，凭先验作答，与 gt（以证据为准 2010）相反；漏读"以证据为准"条件 |
| direct/t33 | 失败（先验答 2014+编造卷期页码） | 维持 | 同上，且编造了具体卷期页码作为"核对信息" |
| direct/t37 | 失败（headline "很可能没超过 5000"） | 维持 | 有审慎分析但 headline 给出判定式结论而非拒绝下数字结论，同 react/t29 标准 |
| direct/t41 | 失败（先验答 8 人） | 维持 | 同 t32 |
| react/t32 | 失败（判检索环境被污染，答 2012） | 维持 | 按任务契约（以证据为准）判失败。**附注**：Agent 识别出"多查询返回逐字相同内容"的模拟环境痕迹，推理本身很敏锐——这揭示了证据忠诚度与污染检测之间的真实张力，记入已知限制 |
| react/t39 | 失败（承认证据缺失仍判"否"） | 维持 | 同 react/t29 的 headline 标准 |
| react/t41 | 失败（先验覆盖证据，答 8 人） | 维持 | 同 react/t32 |
| reflexion/t32 | 失败（以 citation_check 误报为由否弃一致证据，答 2012） | 维持 | 工具误报 + 先验覆盖的复合失效，正是 v1 发现的系统性弱点 |
| reflexion/t41 | 失败（同上模式，答 8 人） | 维持 | 同上 |

无改判。v2 新增失败 9 条全部成立。

## v2 仲裁后基线失败全集（11 条）

react/t29、reflexion/t28（v1 遗留）+ direct/t32、direct/t33、direct/t37、direct/t41、
react/t32、react/t39、react/t41、reflexion/t32、reflexion/t41（v2 新增）。
