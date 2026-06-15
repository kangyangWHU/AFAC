"""Planner：把 profile 转成第一版实验方案（round-0）。
优先用 Qwen 推理；无 key 时走确定性的「画像条件先验」后备。
每条决策都附带触发它的 profile 信号（signal 字段），形成可审计的因果链——
这正是赛题要的「Agent 读数据→决定方案」，且不含写死的最优答案。"""
import os, yaml

_LIB = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "model_library.yaml")))


def _d(model, priority, signal, rationale, space_hint=None):
    return dict(model=model, priority=priority, signal=signal,
               rationale=rationale, space_hint=space_hint or {})


# ---------------- 确定性后备：画像条件先验 ----------------
def _plan_classification_rule(p):
    dec = []
    homo = p.get("edge_homophily") or 0
    iso = p.get("isolated_ratio") or 0
    imb = p.get("imbalance_ratio") or 1
    cw = "balanced" if imb >= 3 else None

    # backbone：孤立点和无标签邻居的测试点只能靠特征，特征模型必入
    dec.append(_d("lightgbm", 1,
                  f"isolated_ratio={iso}, test_with_labeled_neighbor={p.get('test_with_labeled_neighbor_ratio')}",
                  "约1/5节点无边、部分测试点无带标签邻居，需强特征 backbone", {"class_weight_hint": cw}))
    dec.append(_d("feature_mlp", 2, f"feat_nnz_per_node={p.get('feat_nnz_per_node')}",
                  "特征较稠密，MLP 作为第二 backbone 与 LGBM 对比", {"class_weight": cw}))

    # 后处理：homophily 高则 C&S 是最大杠杆
    if homo and homo >= 0.6:
        dec.append(_d("correct_and_smooth", 1, f"edge_homophily={homo}",
                      "相连节点79%同类，标签传播修正几乎免费且强，优先叠加在 backbone 上"))
    if iso and iso >= 0.1:
        dec.append(_d("feature_propagation", 3, f"isolated_ratio={iso}",
                      "孤立节点多，用图传播填补其特征以改善 backbone 输入"))
    # GNN 作为对照组（度低时不押注）
    gnn_prio = 3 if (p.get("avg_degree") or 0) >= 5 else 4
    dec.append(_d("gcn", gnn_prio, f"avg_degree={p.get('avg_degree')}, median_degree={p.get('median_degree')}",
                  "平均度低，GNN 未必胜过 MLP+C&S；作为对照组验证是否有增量"))

    anchor = p.get("val_feature_only_LR_acc")
    return dict(
        decisions=sorted(dec, key=lambda x: x["priority"]),
        round0_feedback={"feature_only_LR_val_acc": anchor},
        first_trial="lightgbm(class_weight=balanced) + correct_and_smooth" if (homo or 0) >= 0.6
                    else "lightgbm(class_weight=balanced)",
        budget={"strategy": "successive_halving",
                "screen_cheap_first": ["feature_lr", "feature_mlp", "lightgbm"],
                "promote_to_full": "top-2 by val acc, then add C&S",
                "gnn_quota": "≤20% 预算，仅作对照"},
        stopping_policy={"min_gain": 0.002, "patience_rounds": 3,
                        "reserve_for_final_retrain_sec": 600},
    )


def _plan_recommendation_rule(p):
    dec = []
    rep = p.get("repeat_rate") or 0
    cold = p.get("cold_start_share") or 0
    empty = p.get("empty_test_share") or 0
    nitems = p.get("n_items") or 0

    dec.append(_d("popularity", 1, f"cold_start_share={cold}, empty_test_share={empty}",
                  "近半测试用户序列≤2、部分为空，热门是冷启动/空序列的兜底"))
    if rep >= 0.3:
        dec.append(_d("repeat_freq", 1, f"repeat_rate={rep}, target_eq_mostfreq={p.get('target_eq_mostfreq')}",
                      "复购率高，按历史频次排序是主信号"))
    dec.append(_d("covisitation", 2, f"repeat_rate={rep}",
                  f"约{round((1-rep)*100)}%为非复购，用共现召回补齐"))
    dec.append(_d("blend", 1, "repeat+pop+covis 组合",
                  "三者加权融合+去重+热门回填，作为综合主力"))
    if nitems and nitems >= 5000:
        dec.append(_d("item_content_knn", 2, f"n_items={nitems}",
                      "item 规模大，用内容相似补长尾召回"))
    # SASRec / MF 作为对照（冷启动占比高时降级）
    seq_prio = 4 if cold >= 0.3 else 3
    dec.append(_d("sasrec", seq_prio, f"cold_start_share={cold}, test_seq_median={p.get('test_seq_len',{}).get('median')}",
                  "测试序列极短，序列模型大概率无增量；作对照组验证后再决定是否放弃"))

    return dict(
        decisions=sorted(dec, key=lambda x: x["priority"]),
        round0_feedback={"popularity_val_ndcg": p.get("val_popularity_ndcg"),
                        "repeat_plus_pop_val_ndcg": p.get("val_repeat_plus_pop_ndcg")},
        first_trial="blend(repeat_freq + covisitation + popularity)",
        budget={"strategy": "cheap_rules_first_then_confirm_heavy",
                "screen_cheap_first": ["popularity", "repeat_freq", "covisitation", "blend"],
                "heavy_quota": "≤25% 预算给 sasrec/mf 做对照"},
        stopping_policy={"min_gain": 0.003, "patience_rounds": 3,
                        "reserve_for_final_retrain_sec": 600},
    )


# ---------------- Qwen 入口（有 key 时） ----------------
def _ask_qwen(profile):
    from . import llm
    if not llm.available():
        return None
    task = profile["task"]
    lib = _LIB[task]
    system = ("你是自动化实验智能体的规划器。根据数据画像和模型库（仅含可选项与超参边界），"
             "输出第一版实验方案。只能从模型库里选模型，不得编造。必须为每个决策标注"
             "依据的画像信号(signal)。严禁直接给出针对该数据的最优超参组合——只给起点与搜索方向。"
             "输出 JSON：{decisions:[{model,priority,signal,rationale,space_hint}],first_trial,budget,stopping_policy}")
    import json
    user = f"数据画像:\n{json.dumps(profile, ensure_ascii=False)}\n\n模型库:\n{json.dumps(lib, ensure_ascii=False)}"
    out = llm.ask_json(system, user)
    if not out:
        return None
    plan, meta = out
    plan["_llm_meta"] = {k: v for k, v in meta.items() if k != "raw"}
    return plan


def make_initial_plan(profile):
    """主入口：返回第一版方案 dict（含 source 标记）。"""
    plan = _ask_qwen(profile)
    if plan is not None:
        plan["source"] = "qwen"
        return plan
    fn = _plan_classification_rule if profile["task"] == "classification" else _plan_recommendation_rule
    plan = fn(profile)
    plan["source"] = "deterministic_fallback"
    return plan
