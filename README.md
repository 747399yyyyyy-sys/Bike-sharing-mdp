#  Bike-sharing Operation Optimization via MDP & Reinforcement Learning / 共享单车运营优化

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg) ![MDP](https://img.shields.io/badge/Model-Markov%20Decision%20Process-orange) ![RL](https://img.shields.io/badge/Algorithm-Reinforcement%20Learning-green) ![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

##  Project Overview | 项目概述
This project builds a data-driven optimization framework for bike-sharing operations under uncertain demand. The system is modeled as a Markov Decision Process (MDP) and solved using Value Iteration and Q-learning to optimize long-term operational performance.

本项目针对共享单车不确定需求下的运营优化问题构建数据驱动决策框架，采用MDP建模并使用值迭代与Q-learning进行求解。

---

##  System Architecture | 系统架构
![Architecture](https://via.placeholder.com/900x300.png?text=MDP+System+Architecture)

Raw Data → Cleaning → Feature Engineering → MDP → Policy Optimization → Evaluation  
原始数据 → 数据清洗 → 特征工程 → MDP建模 → 策略优化 → 评估

---

##  MDP Framework | 决策框架
![MDP](https://via.placeholder.com/900x300.png?text=Markov+Decision+Process+Framework)

State S: weather, season, weekday, demand history  
Action A: {-300, -150, 0, 150, 300}  
Reward R: revenue − idle cost − shortage penalty − service penalty  
γ = 0.95  

状态S：天气/季节/星期/历史需求  
动作A：投放调整  
奖励R：收益-成本-惩罚  
折扣因子γ=0.95  

---

##  Reward Function | 奖励函数
![Reward](https://via.placeholder.com/900x300.png?text=Reward+Function+Structure)

Reward includes:
- Revenue from rentals
- Idle cost
- Shortage penalty
- Service level constraint (≥90%)

奖励包括：
- 租赁收入
- 闲置成本
- 缺车惩罚
- 服务水平约束

---

##  Key Results | 结果分析
![Result](https://via.placeholder.com/900x300.png?text=Policy+Performance+Comparison)

- Value Iteration converges stably  
- Q-learning achieves similar optimal policy  
- Policy consistency ≈ 87%  
- Strong robustness under weather variation  

- 值迭代稳定收敛  
- Q-learning接近最优策略  
- 策略一致性约87%  
- 对天气变化具有鲁棒性  

---

##  Policy Behavior | 策略表现
![Policy](https://via.placeholder.com/900x300.png?text=Policy+Heatmap+or+Decision+Map)

High demand → increase deployment  
Low demand → reduce deployment  
Extreme weather → conservative strategy  

高需求 → 增加投放  
低需求 → 减少投放  
极端天气 → 保守策略  

---

## Repository Structure | 项目结构
```
 Bike-sharing-MDP
 ┣ main_mdp.py
 ┣ paper.pdf
 ┣ data/
 ┃ ┗ day.csv
 ┣ results/
 ┃ ┣ mdp_policy_vi.csv
 ┃ ┣ mdp_policy_qlearning.csv
 ┃ ┗ mdp_optimal_policy_by_date.csv
``` id="z9k3lm"

---

## Tech Stack | 技术栈
Python · NumPy · Pandas · MDP · Reinforcement Learning · Dynamic Programming  

---

## Key Contributions | 主要贡献
MDP formulation, reward design, dual RL implementation, policy validation, state discretization  
MDP建模、奖励函数设计、双强化学习实现、策略验证、状态空间离散化  

---

## AI Usage | AI使用说明
Used only for formatting and documentation assistance. All modeling and analysis were independently completed.  
仅用于格式优化与文本润色，建模与分析均独立完成。

---

## Conclusion | 结论
This project demonstrates how reinforcement learning and MDP can be applied to real-world operational optimization problems.

本项目展示强化学习与MDP在真实运营优化问题中的应用。
