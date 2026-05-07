import pandas as pd
import numpy as np
from collections import defaultdict

df = pd.read_csv("day.csv")
df["dteday"] = pd.to_datetime(df["dteday"])
df.drop_duplicates(inplace=True)

# 缺失值直接用每一行的均值替代
num_cols = ["temp", "atemp", "hum", "windspeed"]
for c in num_cols:
    if df[c].isna().any():
        df[c] = df[c].fillna(df[c].mean())

df = df.dropna(subset=["cnt"])

# IQR法异常值处理
def clean_outliers(data, cols):
    res = data.copy()
    for c in cols:
        q1 = res[c].quantile(0.25)
        q3 = res[c].quantile(0.75)
        iqr = q3 - q1
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        res = res[(res[c] >= low) & (res[c] <= high)]
    return res

df = clean_outliers(df, num_cols + ["cnt"])

# 考虑模型的滞后性
df = df.sort_values("dteday")
df["prev_cnt"] = df["cnt"].shift(1).bfill()#前一天的
df["prev_week_cnt"] = df["cnt"].shift(7).bfill()#前一个星期的

# 需求等级分成三位
df["demand_level"] = pd.qcut(df["cnt"], 3, labels=False)
df["prev_demand"] = pd.qcut(df["prev_cnt"], 3, labels=False)
df["prev_week_demand"] = pd.qcut(df["prev_week_cnt"], 3, labels=False)

# 天气，温度，湿度离散
df["weather_simplified"] = df["weathersit"].replace({1: 0, 2: 1, 3: 2, 4: 2})#天气情况四类转换
df["temp_level"] = pd.qcut(df["temp"], 3, labels=False)#温度分三档
df["hum_level"] = pd.qcut(df["hum"], 3, labels=False)#湿度分三档

df["date_str"] = df["dteday"].dt.strftime("%Y-%m-%d")#日期格式化

mdp_df = df[
    ["date_str", "season", "mnth", "weekday", "workingday",
     "weather_simplified", "demand_level",
     "prev_demand", "prev_week_demand",
     "temp_level", "hum_level", "cnt"]
]

mdp_df.to_csv("cleaned_day_mdp.csv", index=False)#得到第一个呈现，清洗后的数据
print('数据清洗完成')

# 4. MDP 状态空间与转移概率
#状态空间
df = pd.read_csv("cleaned_day_mdp.csv")
df["date_str"] = pd.to_datetime(df["date_str"])

STATE_COLS = [
    "season", "mnth", "weekday", "workingday",
    "weather_simplified",
    "demand_level", "prev_demand", "prev_week_demand",
    "temp_level", "hum_level"
]

# 状态映射（把所有需要用到的量打包成一个向量）
state_df = df[STATE_COLS].drop_duplicates().reset_index(drop=True)
state_df["state_id"] = state_df.index

df = df.merge(state_df, on=STATE_COLS, how="left")
num_states = len(state_df)

# 马尔科夫链（状态转移）
df = df.sort_values("date_str")
trans_cnt = defaultdict(lambda: defaultdict(int))

for i in range(len(df) - 1):
    s = df.iloc[i]["state_id"]
    s2 = df.iloc[i + 1]["state_id"]
    trans_cnt[s][s2] += 1#每一种状态改变方式，每出现一次就加一

P = {}
for s in trans_cnt:
    tot = sum(trans_cnt[s].values())
    P[s] = {s2: c / tot for s2, c in trans_cnt[s].items()}#状态出现的概率
print('状态转移成功')

# 5. 动作空间、奖励函数与值迭代（模型核心）

# 每个状态下的历史需求样本
need_samples = {
    sid: g["cnt"].values
    for sid, g in df.groupby("state_id")
}
#动作空间
# 95百分位法则，去除极端值（敏感性分析1）
service_quantile = 0.95
base_deploy = {
    sid: int(np.quantile(need_samples[sid], service_quantile))
    for sid in need_samples
}

# 在基础值附近微调（敏感性分析2）
action_shift = [-300, -150, 0, 150, 300]
A = {
    sid: sorted({max(base_deploy[sid] + d, 0) for d in action_shift})
    for sid in base_deploy
}#储存每一种可能性（为了后面贝尔曼方程）

#奖励函数
reward_params = {
    "price": 1.0,
    "idle_cost": 0.1,
    "shortage_cost": 2.0,
    "service_target": 0.9,
    "penalty": 5.0
}

def calc_reward(sid, a):
    D = need_samples[sid]

    served = np.minimum(a, D)#需求量
    idle = np.maximum(a-D,0) #闲置量
    shortage = np.minimum(a-D,0) #缺少量

    fill_rate = served.sum() / D.sum()#用户满意百分值

    r = (
        reward_params["price"] * served.mean()
        - reward_params["idle_cost"] * idle.mean()
        - reward_params["shortage_cost"] * shortage.mean()
    )

    if fill_rate < reward_params["service_target"]:
        r -= reward_params["penalty"] * (reward_params["service_target"] - fill_rate)#收益惩罚制度

    return r
print('奖励函数结束')

#值迭代
gamma = 0.95
max_iter = 200#迭代最大次数
edge= 1e-4

V = np.zeros(num_states)
policy = np.zeros(num_states, dtype=int)

for _ in range(max_iter):
    diff = 0
    for s in range(num_states):
        v0 = V[s]
        best_v = -1e18
        best_a = None#初始化

        for a in A[s]:
            future = sum(P.get(s, {}).get(s2, 0) * V[s2] for s2 in P.get(s, {}))#算未来的收益
            q = calc_reward(s, a) + gamma * future#贝尔曼=今天的收益+未来收益
            if q > best_v:
                best_v = q
                best_a = a#贪心算法结构辅助

        V[s] = best_v#最佳收益（价值）
        policy[s] = best_a#最佳策略
        diff = max(diff, abs(v0 - V[s]))#取最大值（最优价值，最优动作）

    if diff < edge:
        break

policy_df = state_df.copy()
policy_df["deploy_mdp"] = policy
policy_df.to_csv("mdp_policy_vi.csv", index=False)
print('值迭代完成')

#qlearning（尝试优化）
import random

Q = {s: {a: 0.0 for a in A[s]} for s in range(num_states)}#原始创建q table

alpha = 0.1
gamma = 0.95
epsilon = 0.2#经典ε-greedy，没用衰减ε-greedy
episodes = 50

df_rl = df.sort_values("date_str").reset_index(drop=True)

def choose_action(s):
    if random.random() < epsilon:
        return random.choice(A[s])
    return max(Q[s], key=Q[s].get)

for _ in range(episodes):
    for i in range(len(df_rl) - 1):
        s = df_rl.loc[i, "state_id"]
        s2 = df_rl.loc[i + 1, "state_id"]

        a = choose_action(s)
        r = calc_reward(s, a)

        Q[s][a] += alpha * (r + gamma * max(Q[s2].values()) - Q[s][a])

rl_policy = {s: max(Q[s], key=Q[s].get) for s in Q}

rl_df = state_df.copy()
rl_df["deploy_rl"] = rl_policy.values()
rl_df.to_csv("mdp_policy_qlearning.csv", index=False)

# 6. 最优策略映射到具体日期

df_policy = df.copy()

df_policy["optimal_deploy"] = df_policy["state_id"].apply(lambda s: policy[s])
df_policy["base_deploy"] = df_policy["state_id"].apply(lambda s: base_deploy[s])
df_policy["adjustment"] = df_policy["optimal_deploy"] - df_policy["base_deploy"]
print('qlearning 完成')

def action_label(diff):
    if diff < -200:
        return "大幅减少投放"
    elif diff < -50:
        return "减少投放"
    elif abs(diff) <= 50:
        return "维持现状"
    elif diff <= 200:
        return "增加投放"
    else:
        return "大幅增加投放"

df_policy["recommended_action"] = df_policy["adjustment"].apply(action_label)
df_policy["state_value"] = df_policy["state_id"].apply(lambda s: round(V[s], 3))

final_cols = [
    "date_str",
    "season","mnth","weekday","workingday",
    "weather_simplified",
    "demand_level",
    "temp_level","hum_level",
    "base_deploy",
    "optimal_deploy",
    "adjustment",
    "recommended_action",
    "state_value"
]

final_policy_df = (
    df_policy[final_cols]
    .drop_duplicates()
    .sort_values("date_str")
)

final_policy_df.to_csv(
    "mdp_optimal_policy_by_date.csv",
    index=False,
    encoding="utf-8-sig"
)
print('代码已全部运行成功')

