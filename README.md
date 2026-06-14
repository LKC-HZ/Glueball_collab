# color_wf: QCD 颜色单态波函数求解器

在 **Cartan-Weyl 约化基**（$I_3 = Y = 0$）下，利用 ARPACK 本征值求解器，
计算由 $n$ 个夸克、$m$ 个反夸克、$k$ 个胶子构成的 **SU(3) 颜色单态波函数**。

## 编译

```bash
gfortran -O2 -o color_wf color_wf.f90 -larpack -llapack -lblas
```

依赖：Fortran 编译器、ARPACK（复数版本 `znaupd`/`zneupd`）、LAPACK、BLAS。

## 使用

```bash
./color_wf <n_quarks> <m_antiquarks> <k_gluons>
```

或无参数运行，程序会交互式提示输入。

### 仅计算单态数目（不计算波函数）

加 `-c` 选项可跳过 ARPACK 求解和波函数输出，仅输出单态数目：

```bash
./color_wf -c <n_quarks> <m_antiquarks> <k_gluons>
```

这对大维度系统（如 `0 0 6`，全空间 $8^6 = 262144$）非常有用，可在不耗费大量计算资源的情况下快速获知单态数量。

### 示例

```bash
./color_wf 3 0 0    # 三夸克（重子）
./color_wf 1 1 0    # 夸克-反夸克（介子）
./color_wf 0 0 3    # 三胶子
./color_wf 1 1 1    # 夸克-反夸克-胶子
./color_wf 0 0 4    # 四胶子
./color_wf -c 0 0 6 # 仅计算六胶子单态数目
```

### 输入约束

- $n, m, k \geq 0$
- **Triality 约束**：$n - m \equiv 0 \pmod{3}$（否则没有颜色单态）
- 总 Hilbert 空间维数 $3^{n+m} \times 8^k$ 不能超过 500,000

## 输出说明

程序同时输出两套基下的波函数：

1. **Cartan-Weyl (CW) 基** — 胶子用 $(I_3, T_8)$ 权向量标记
2. **Gell-Mann (GM) 基** — 胶子用 $\lambda_{1\ldots8}$ 标记

### 粒子标记

| 粒子 | 标记 | 含义 |
|------|------|------|
| 夸克 | `q1, q2, q3` | 基础表示 $\mathbf{3}$ 的三个色态 |
| 反夸克 | `qb1, qb2, qb3` | 反基础表示 $\bar{\mathbf{3}}$ 的三个色态 |
| 胶子 (CW) | `g_C1, g_C2`, `g(+1,0)`, `g(-1/2,+s3/2)`, ... | 见下表 |
| 胶子 (GM) | `lam1, lam2, ..., lam8` | Gell-Mann 矩阵对应的 8 个胶子态 |

### Cartan-Weyl 胶子态

| CW 标记 | GM 定义 | $(I_3, T_8)$ | 类型 |
|---------|---------|--------------|------|
| `g_C1` | $\lambda_3$ | $(0, 0)$ | Cartan |
| `g_C2` | $\lambda_8$ | $(0, 0)$ | Cartan |
| `g(+1,0)` | $(\lambda_1 + i\lambda_2)/\sqrt{2}$ | $(+1, 0)$ | 正根 |
| `g(-1,0)` | $(\lambda_1 - i\lambda_2)/\sqrt{2}$ | $(-1, 0)$ | 负根 |
| `g(+1/2,+s3/2)` | $(\lambda_4 + i\lambda_5)/\sqrt{2}$ | $(+1/2, +\sqrt{3}/2)$ | 正根 |
| `g(-1/2,-s3/2)` | $(\lambda_4 - i\lambda_5)/\sqrt{2}$ | $(-1/2, -\sqrt{3}/2)$ | 负根 |
| `g(-1/2,+s3/2)` | $(\lambda_6 + i\lambda_7)/\sqrt{2}$ | $(-1/2, +\sqrt{3}/2)$ | 正根 |
| `g(+1/2,-s3/2)` | $(\lambda_6 - i\lambda_7)/\sqrt{2}$ | $(+1/2, -\sqrt{3}/2)$ | 负根 |

### d-type / f-type 标签

当存在多个单态时（如 3 胶子、4 胶子），程序通过粒子交换 $P_{12}$ 的本征值来区分：

- **d-type** ($P_{12} = +1$)：交换粒子 1 和粒子 2 后波函数不变
- **f-type** ($P_{12} = -1$)：交换粒子 1 和粒子 2 后波函数变号

对于 3 胶子系统，这对应于 $d^{abc}$（全对称）和 $f^{abc}$（全反对称）耦合结构。

## 算法原理

### 第一步：理论单态计数

通过 SU(3) 表示的张量积规则，逐步计算：

$$\underbrace{\mathbf{3} \otimes \cdots \otimes \mathbf{3}}_{n} \otimes \underbrace{\bar{\mathbf{3}} \otimes \cdots \otimes \bar{\mathbf{3}}}_{m} \otimes \underbrace{\mathbf{8} \otimes \cdots \otimes \mathbf{8}}_{k}$$

中单态出现的次数。用 Dynkin 标记 $(p, q)$ 追踪中间表示的维数，最终读出 $(0,0)$ 分量。

### 第二步：构建零权子空间

全空间维度为 $3^{n+m} \times 8^k$，对大规模问题直接对角化不现实。

利用 **Cartan 算符** $T^3, T^8$（对角）的性质，筛选出 $I_3 = T^8 = 0$ 的基矢，
构建约化子空间。典型情况下，零权子空间远小于全空间。

### 第三步：ARPACK 求解

在约化空间上，构造 **二次 Casimir 算符** $C_2 = \sum_{a=1}^{8} (T^a)^2$ 的矩阵-向量乘积。

颜色单态满足 $C_2 |\psi\rangle = 0$，因此使用 ARPACK（`znaupd`/`zneupd`）
求解最小本征值问题，零本征值对应的本征向量即为单态。

### 第四步：相位固定与正交化

ARPACK 给出的本征向量带有任意全局相位。采用 **CW 相位固定法**：

$$\phi = \frac{1}{2}\arg\!\Bigl(\sum_i \psi_i^2\Bigr), \quad \psi \to e^{-i\phi}\psi$$

这保证所有 CW 系数为实数（色单态的 CW 系数本质上是实的），然后进行实 Gram-Schmidt 正交化。

### 第五步：对称性对角化（可选）

当存在多个单态时，通过以下算符的本征值进一步区分：

- **M 算符**（权反转）：$M = +1$ / $M = -1$，仅当 $n = m$ 时使用
- **粒子交换 $P_{12}$**：$P_{12} = +1$（d-type）/ $P_{12} = -1$（f-type）

用 Jacobi 方法对角化对应的重叠矩阵，将简并单态对角化为确定对称性的态。

### 第六步：GM 基变换

通过 CW→GM 变换矩阵 $U$，对每个胶子粒子做基变换：

$$|\psi_{\mathrm{GM}}\rangle = \prod_{p \in \text{gluons}} U^T |\psi_{\mathrm{CW}}\rangle$$

其中 $U$ 的行就是 CW 态在 GM 基下的展开系数。

## 物理验证

| 输入 | 物理系统 | 单态数 | 波函数 |
|------|----------|--------|--------|
| `3 0 0` | 重子 | 1 | $\epsilon_{ijk}/\sqrt{6}$ |
| `1 1 0` | 介子 | 1 | $\delta_{i\bar{j}}/\sqrt{3}$ |
| `0 0 3` | 三胶子 | 2 | $d^{abc}$ (d-type) + $f^{abc}$ (f-type) |
| `1 1 1` | qq̄g | 1 | $T^a_{ij}/2$ 结构 |
| `0 0 4` | 四胶子 | 8 | $dd, df, fd, ff$ 等收缩结构 |

## 代码结构

```
color_wf.f90
├── 主程序：输入解析 → 构建生成元 → 零权子空间 → ARPACK → 后处理 → 输出
│
└── contains (内部子程序)
    ├── build_cartan_weyl_basis()    — CW 变换矩阵和量子数
    ├── init_generators_GM()         — Gell-Mann 基生成元
    ├── transform_gluon_to_CW()      — GM→CW 生成元变换
    ├── build_zero_weight_subspace_cartan() — 零权子空间 + M 映射
    ├── t_action()                   — 单粒子生成元作用
    ├── c2_matvec_red() / c2_matvec_full() — Casimir 矩阵-向量乘积
    ├── apply_Ta() / apply_M_full()  — 生成元/M 算符作用
    ├── count_singlets()             — 理论单态计数（表示张量积）
    ├── jacobi_diag()                — 小矩阵 Jacobi 对角化
    ├── cw_to_gm_vec() / gm_to_cw_vec() — CW↔GM 基变换
    └── swap_overlap()               — 粒子交换重叠矩阵元
```

## 数值约定

- 双精度浮点（`real64`）
- Casimir 零本征值判定阈值：$10^{-10}$
- 非零系数输出阈值：$10^{-10}$
- 虚部噪声清理：相对虚部 $< 10^{-10}$ 时置零
