"""实验台账（Experiment Memory）：记录每次试验，并产出给 LLM 读的【紧凑摘要】。
任务无关——只认 config(dict) 与 metric(float) 与 slices(dict)，因此换任务零改动。
摘要回答 LLM 三个问题：现在最好多少？试过什么/没试什么？最好方案在哪些切片上烂？"""
import json, time


class ExperimentMemory:
    def __init__(self, task, metric_name, baseline=None):
        self.task = task
        self.metric_name = metric_name
        self.baseline = baseline
        self.trials = []

    def add(self, config, metric, slices=None, cost_sec=None,
            status="ok", decided_by=None, signal=None):
        self.trials.append(dict(
            id=len(self.trials), config=config, metric=metric,
            slices=slices or {}, cost_sec=cost_sec, status=status,
            decided_by=decided_by, signal=signal, ts=time.time()))
        return self.trials[-1]

    def best(self):
        ok = [t for t in self.trials if t["status"] == "ok"]
        return max(ok, key=lambda t: t["metric"]) if ok else None

    def _coverage(self):
        """每个超参各试过哪些取值 —— 让 LLM 知道搜索空间哪里还没探。"""
        cov = {}
        for t in self.trials:
            for k, v in (t["config"] or {}).items():
                cov.setdefault(k, {})
                key = str(v)
                cov[k][key] = cov[k].get(key, 0) + 1
        return cov

    def digest(self, top_k=5, recent=5):
        """喂给 LLM 的紧凑状态（不是原始日志，省 token）。"""
        ok = [t for t in self.trials if t["status"] == "ok"]
        top = sorted(ok, key=lambda t: -t["metric"])[:top_k]
        best = self.best()
        return dict(
            task=self.task, metric=self.metric_name,
            baseline=self.baseline,
            n_trials=len(self.trials),
            n_failed=sum(t["status"] != "ok" for t in self.trials),
            best_metric=best["metric"] if best else None,
            leaderboard=[dict(id=t["id"], config=t["config"],
                             metric=round(t["metric"], 4)) for t in top],
            recent_deltas=[round(t["metric"] - (best["metric"] if best else 0), 4)
                          for t in self.trials[-recent:]],
            coverage=self._coverage(),
            best_weak_slices=self._weak_slices(best) if best else None,
            failures=[dict(id=t["id"], config=t["config"], status=t["status"])
                     for t in self.trials if t["status"] != "ok"][-3:],
        )

    @staticmethod
    def _weak_slices(trial, k=4):
        sl = trial.get("slices") or {}
        items = [(name, d) for name, d in sl.items() if isinstance(d, dict) and "metric" in d]
        return [dict(slice=n, **d) for n, d in sorted(items, key=lambda x: x[1]["metric"])[:k]]

    def to_trajectory(self):
        """复审用的过程日志（trajectory_B*.json 的内容）。"""
        return dict(task=self.task, metric=self.metric_name,
                   baseline=self.baseline, rounds=self.trials)

    def save(self, path):
        json.dump(self.to_trajectory(), open(path, "w"), ensure_ascii=False, indent=2)
