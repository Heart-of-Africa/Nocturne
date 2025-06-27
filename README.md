# Nocturne
为了更好地理解本项目中所融合的机制，以下简要回顾了 **液体神经网络（Liquid Neural Networks, LNN）** 与 **马尔可夫链（Markov Chain）** 的核心概念：

### 液体神经网络（Liquid Neural Networks）

- 液体神经网络是一种基于微分方程描述神经元动态的网络结构。
- 不同于传统神经网络的离散更新方式，LNN 使用连续时间建模，其动态表达为：d h(t) / dt = f(h(t), u(t); θ)
-  通常使用数值方法（如 Euler 或 Runge-Kutta）来求解状态随时间的演化。
- 其核心优势在于更高的动态适应性和对时间变化的敏感性。

### 马尔可夫链（Markov Chain）

- 马尔可夫链是指具有**无记忆性**的随机过程，即状态的转移只依赖于当前状态。
- 数学定义如下：P(X_{t+1} | X_t, X_{t-1}, ..., X_0) = P(X_{t+1} | X_t)

- 通常通过状态转移矩阵 P 来描述整个过程，适合建模序列中具有局部依赖性或跳跃结构的问题。
- 在神经网络中，马尔可夫链可以用于描述状态之间的**非确定性跳跃式传播**，例如用于改造反向传播路径。


## 项目目标与创新

本项目尝试构造一种**马尔可夫式反向传播（Markovian Backpropagation）**，在传统链式法则的基础上引入状态跳跃概率控制，使误差信号传播具有：

- 更强的鲁棒性（尤其在非平稳任务中）
- 更强的时序建模能力
- 可解释性更强的状态转移路径

此外，我们还探索了 LNN 单元与马尔可夫链的结构融合，以改造梯度传播路径并减少局部收敛陷阱。

---

## 方法概述

### 1. 正向传播阶段（Forward Pass）

- 使用液体神经元单元构造隐藏层，基于微分方程求解状态随时间变化。
- 每一层输出 \( h_t \) 作为下层输入，同时被用于构建状态转移矩阵 \( M_t \)。

### 2. 马尔可夫反向传播（Backward Phase）

- 不再使用标准链式法则，而使用以下方式传播梯度：grad_h[t] = M_t @ grad_h[t+1]
- \( M_t \) 可由液体神经元状态构造（例如 outer product 后 softmax 归一化），模拟跳跃式传播。

---

##  安装与运行

```bash
git clone https://github.com/Heart-of-Africa/Nocturne
cd liquid-markov-backprop
pip install -r requirements.txt
python train.py --config configs/example.yaml
