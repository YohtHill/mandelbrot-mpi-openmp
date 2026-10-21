# Mandelbrot：MPI + OpenMP 混合并行渲染器

[English](README.md) | [简体中文](README.zh-CN.md)

这是一个用于 Lab 9 期末并行编程挑战的 MPI + OpenMP 混合渲染器。图像按行带划分给各个 MPI rank；每个 rank 用 OpenMP 线程渲染自己的行带；rank 0 用 `MPI_Gatherv` 汇总所有行带，并写出二进制 PPM 图像。

项目比较静态和动态 OpenMP 调度，验证所有配置产生完全相同的图像，并用 Mandelbrot 各行计算量不均匀这一事实解释测得的加速比。

![Mandelbrot 渲染图](docs/mandelbrot.png)

## 1. 程序做什么

对每个像素，渲染器会：

1. 把像素中心映射为复数 `c = cr + i*ci`；
2. 从 `z = 0` 开始；
3. 重复 `z = z*z + c`，直到 `|z| > 2` 或达到 `max_iter`；
4. 根据像素坚持的迭代次数决定颜色；
5. 把 RGB 字节写入二进制 PPM 文件；
6. 计算整幅图像的 FNV-1a 哈希。

测量的图像尺寸是 **800 x 600**，**max_iter = 1000**。最终 PPM 为 1,440,015 字节：1,440,000 个 RGB 字节，加上 15 字节的 `P6` 文件头。

程序是确定性的。每种配置都必须打印：

```text
FNV-1a: 92193b63bbd42a23
```

并且每个 PPM 都必须有相同的 SHA-256：

```text
4606e1e408496266fd928852c8b0c4137113773c7d01f4b814b3ad058827c17e
```

## 2. Mandelbrot 细节

### 2.1 复平面窗口

图像使用：

```text
xmin = -2.10
xmax =  0.70
xspan = xmax - xmin
yspan = xspan * height / width
ymin = -0.5 * yspan
ymax =  0.5 * yspan
```

纵向范围由横向范围推导而来，以保持像素为正方形。图像中心对应复平面原点。

### 2.2 像素到复数

采样的是像素中心，而不是像素角点：

```text
cr = xmin + (xmax - xmin) * (px + 0.5) / width
ci = ymax - (ymax - ymin) * (py + 0.5) / height
```

### 2.3 逃逸时间循环

```c
zr = 0.0;
zi = 0.0;
iter = 0;

while (iter < max_iter && zr*zr + zi*zi <= 4.0) {
    next_r = zr*zr - zi*zi + cr;
    next_i = 2.0*zr*zi + ci;
    zr = next_r;
    zi = next_i;
    iter++;
}
```

达到 `max_iter` 的像素被视为在集合内部，涂成黑色。逃逸的像素根据其迭代次数着色。

### 2.4 配色

对于逃逸的像素：

```text
t = sqrt(iter / 200.0)          // 限制到 1.0
R = 255 * min(1, 2t)
G = 255 * min(1, 2t - 0.5)
B = 255 * min(1, 1 - 2t)
```

平方根拉伸了低迭代次数区域，所以外围不会只是一整片平色。配色不影响并行行为，只是让计算昂贵的边界区域更容易看出来。

## 3. 串行参考实现

`src/mandelbrot_serial.c` 是参考实现。它在一个进程中渲染整幅图像，写出 PPM，并打印耗时和 FNV 哈希。

测得的串行时间：

```text
第 1 次：0.729 s
第 2 次：0.723 s
第 3 次：0.723 s
平均  ：0.725 s
```

串行时间只测量逃逸时间循环，不包含最后写 PPM 和计算哈希。并行时间包含 `MPI_Gatherv`，所以报告中的并行加速比略偏保守。

## 4. MPI 行分解

### 4.1 划分公式

对于 `P` 个 MPI rank 和 `H` 行图像：

```text
base = H / P
rem  = H % P
local_h = base + (rank < rem ? 1 : 0)
y0      = rank * base + min(rank, rem)
```

前 `rem` 个 rank 各多分一行。对于 `H = 600`、`P = 4`：

| Rank | y0 | 行数 | 结束行 |
|---|---:|---:|---:|
| 0 | 0 | 150 | 150 |
| 1 | 150 | 150 | 300 |
| 2 | 300 | 150 | 450 |
| 3 | 450 | 150 | 600 |

每个 rank 分配一个紧凑缓冲区：

```text
local_h * width * 3 字节
```

并且只渲染属于自己的行。

### 4.2 不需要 halo 交换

每个输出像素只属于一个 rank。没有邻居数据、没有幽灵行，也不需要 rank 之间的迭代。唯一的 MPI 通信是：

1. 三次小规模广播，分别发送 `width`、`height` 和 `max_iter`；
2. 最后的一次 `MPI_Gatherv`。

这就是通信开销相对渲染计算很小的原因。

### 4.3 广播参数

Rank 0 解析命令行，并广播三个整数：

```c
MPI_Bcast(&width,    1, MPI_INT, 0, MPI_COMM_WORLD);
MPI_Bcast(&height,   1, MPI_INT, 0, MPI_COMM_WORLD);
MPI_Bcast(&max_iter, 1, MPI_INT, 0, MPI_COMM_WORLD);
```

所有 rank 必须对图像尺寸和迭代上限达成一致；否则缓冲区和行划分会对不上。

### 4.4 汇总图像

渲染完成后，rank 0 用 `MPI_Gatherv` 接收所有行带。

对于每个 rank `r`，代码计算：

```text
rows_r    = base + (r < rem ? 1 : 0)
counts[r] = rows_r * width * 3
displs[r] = sum of counts[0..r-1]
```

`counts` 的单位是 **字节**，不是行数，因为 `MPI_Gatherv` 统计的是发送缓冲区中的元素个数，而每个元素是一个 RGB 字节。`displs` 的单位也是字节，指向该行带在最终图像中的起始位置。

根进程分配完整图像：

```c
full = malloc(width * height * 3);
```

然后调用：

```c
MPI_Gatherv(local, local_bytes, MPI_UNSIGNED_CHAR,
            full, counts, displs, MPI_UNSIGNED_CHAR,
            0, MPI_COMM_WORLD);
```

之后 rank 0 计算 FNV 哈希、写出 PPM，并打印汇总信息。

### 4.5 计时

所有 rank 在 `t0 = MPI_Wtime()` 之前调用 `MPI_Barrier`。计时区间包含：

- 本地渲染；
- `MPI_Gatherv`。

不包含最后的 PPM 写入。墙钟时间取 rank 0 的 `t1 - t0`。

## 5. OpenMP 层

在一个 rank 内部，`render_rows` 并行化行循环：

```c
#pragma omp parallel for
for (int j = 0; j < nrows; ++j) {
    ...
}
```

每个线程有自己的 `j`，计算自己的 `ci`，并写入本地缓冲区中互不重叠的部分。没有共享累加器，也没有归约，所以这个循环没有数据竞争。

### 5.1 静态调度

```c
#pragma omp parallel for schedule(static)
```

迭代在开始时被划分为连续块。对于 600 行和 4 个线程，每个线程分到 150 行。

### 5.2 动态调度

```c
#pragma omp parallel for schedule(dynamic, 8)
```

线程一次领取 8 行，做完后再领取下一个 8 行。这样，分到便宜行的线程可以去接管中间计算昂贵的区域。

8 行是一个折中：

- 块越小，负载越均衡，但调度开销越大；
- 块越大，调度开销越小，但最后可能留下很长的尾巴。

## 6. 确定性与浮点

渲染器没有随机数生成器。一个像素的颜色只由它的坐标和 `max_iter` 决定，因此线程或 rank 处理像素的顺序不会改变结果。

构建时使用：

```text
-ffp-contract=off
```

这会阻止编译器把一次乘法和一次加法融合成 FMA 指令。没有这个选项时，两次构建可能在边界像素上相差一次迭代，从而改变颜色和哈希，即使算法完全相同。

## 7. 构建

要求：

- Ubuntu Server 24.04（或其他带 GCC 和 OpenMPI 的 Linux）；
- 支持 OpenMP 的 GCC；
- OpenMPI（`mpicc`、`mpirun`）；
- 可选：ImageMagick 和 `chafa`，用于在终端查看 PPM。

安装：

```bash
sudo apt update
sudo apt install build-essential libopenmpi-dev openmpi-bin
```

构建：

```bash
make
```

Makefile 会生成：

```text
mandelbrot_serial
mandelbrot_static
mandelbrot_dynamic
```

重要编译选项：

```text
-O2 -std=c11 -Wall -Wextra -fopenmp -ffp-contract=off -lm
```

## 8. 运行

### 8.1 完整实验

```bash
./scripts/run.sh
```

脚本会运行：

- serial 1x1；
- static 1x4 和 2x2；
- dynamic 1x4、2x2 和 4x1；

然后打印所有输出图像的 SHA-256。

### 8.2 手动运行一个配置

```bash
OMP_NUM_THREADS=2 mpirun --bind-to none -x OMP_NUM_THREADS \
  -np 2 ./mandelbrot_dynamic 800 600 1000 out/2x2.ppm
```

`--bind-to none` 防止 OpenMPI 把一个 rank 的所有线程都绑定到一个核上。`-x OMP_NUM_THREADS` 会把线程数转发给每个 rank。

### 8.3 命令行

```text
./mandelbrot_dynamic [width] [height] [max_iter] [output.ppm]
```

| 参数 | 默认值 | 含义 |
|---|---:|---|
| width | 800 | 图像宽度（像素） |
| height | 600 | 图像高度（像素） |
| max_iter | 1000 | 迭代上限 |
| output.ppm | mandelbrot.ppm | 输出文件 |

`mandelbrot_serial` 和 `mandelbrot_static` 接受相同的参数。

### 8.4 输出

每个 rank 先打印自己的行范围，然后 rank 0 打印汇总：

```text
rank 0/4: y0=0 rows=150
rank 1/4: y0=150 rows=150
...
MPI+OpenMP Mandelbrot 800x600 max_iter=1000 ranks=4 threads/rank=1
time        : 0.310 s
FNV-1a hash : 92193b63bbd42a23
output      : out/4x1_dynamic.ppm
```

`time` 包含渲染和 `MPI_Gatherv`，不包含写文件。哈希由 rank 0 从汇总后的图像计算。

## 9. 结果

在 VMware Fusion 的 Ubuntu Server 24.04 虚拟机上测得，**4 个 vCPU**，800 x 600，`max_iter = 1000`，另外还在实验室集群上做了两节点检查。

### 9.1 静态基线

这些是单次观测，用作静态对比。

| 配置 | Worker 数 | 时间（s） | 加速比 | 效率 |
|---|---:|---:|---:|---:|
| serial 1x1 | 1 | 0.725 | 1.00 | 1.000 |
| static 1x4 | 4 | 0.309 | 2.35 | 0.587 |
| static 2x2 | 4 | 0.326 | 2.22 | 0.556 |
| static 4x1 | 4 | 0.310 | 2.34 | 0.585 |

### 9.2 动态调度

这些是三次运行的平均值。

| 配置 | 第 1 次 | 第 2 次 | 第 3 次 | 平均 | 加速比 | 效率 |
|---|---:|---:|---:|---:|---:|---:|
| dynamic 1x4 | 0.183 | 0.183 | 0.181 | 0.1823 | 3.98 | 0.994 |
| dynamic 2x2 | 0.184 | 0.182 | 0.184 | 0.1833 | 3.95 | 0.989 |
| dynamic 4x1 | 0.309 | 0.310 | 0.309 | 0.3093 | 2.34 | 0.586 |

加速比和效率定义为：

```text
S(W) = T(1) / T(W)
E(W) = S(W) / W
```

其中 `W` 是 worker 总数（rank 数乘以每个 rank 的线程数）。串行平均值是 `T(1)` 基准。

### 9.3 两节点检查

同一组二进制程序也在两个计算节点上运行过。第一种配置是每个节点 1 个 rank、2 个 OpenMP 线程；第二种是每个节点 2 个 rank、每个 rank 1 个线程。

| 配置 | 布局 | Worker 数 | 平均时间（s） | 加速比 | 效率 |
|---|---|---:|---:|---:|---:|
| headnode 上的 serial | 1 rank，1 线程 | 1 | 0.416 | 1.00 | 1.000 |
| 跨两节点的 2x2 | 每节点 1 rank，2 线程 | 4 | 0.1193 | 3.49 | 0.872 |
| 跨两节点的 4x1 | 每节点 2 rank，1 线程 | 4 | 0.1977 | 2.10 | 0.526 |

2x2 混合版本比 4x1 快 1.66 倍，尽管两者都使用 4 个 worker。这重复了单机上的规律：混合版本让每个 rank 有 2 个线程，可以分担昂贵的行；而 4x1 每个 rank 只有 1 个线程，仍然要承受静态 MPI 行带划分带来的不均衡。所有运行都得到相同的 FNV 哈希。

多节点时间不与单 VM 的 0.725 s 基准直接比较：计算节点和虚拟机的 CPU 分配、负载都不同。在集群环境中，0.416 s 的串行平均值才是正确的基准。

## 10. 为什么结果是这样

### 10.1 工作量不均匀

穿过图像中心的行会穿过 Mandelbrot 集合，需要跑到 `max_iter`。靠近顶部和底部的行几乎立刻逃逸。

对同一幅 800 x 600 图像做逐行开销测量，得到：

- 中心行（第 300 行）：平均约 **612 次迭代/像素**；
- 外围行：平均约 **3 次迭代/像素**。

两者相差超过两个数量级。

因此图像不是一个均匀负载。如果调度器给每个 worker 固定的连续块，就可能让某个 worker 拿到大部分昂贵行。

### 10.2 为什么静态调度停在约 2.3x

三种静态配置的粒度都是 150 行：

| 配置 | 600 行如何划分 | 最小静态块 |
|---|---|---:|
| static 1x4 | 600 / 4 个线程 | 150 行 |
| static 2x2 | 600 / 2 个 rank，再 300 / 2 个线程 | 150 行 |
| static 4x1 | 600 / 4 个 rank | 150 行 |

最慢的 150 行块包含昂贵的中间行，耗时约 0.31 s。其他 worker 更早完成，然后等待。墙钟时间由这个块决定：

```text
0.725 / 0.310 ≈ 2.34x
```

这解释了为什么所有静态配置都在 2.2–2.35x 左右，即使它们使用了 4 个 worker。

static 2x2 的结果（0.326 s）比 1x4（0.309 s）略差，因为两个 300 行的行带仍然存在 rank 级划分，最后的 gather 也增加了一点开销。差异很小。

### 10.3 为什么动态调度达到约 3.98x

使用 `schedule(dynamic, 8)` 时，空闲线程会立刻再领取 8 行。昂贵的行不再由某一个线程在整个运行期间独占，而是由空闲下来的线程共同分担。

4 个 worker 的理想时间是：

```text
0.725 / 4 = 0.181 s
```

测得的 1x4 平均值为 0.1823 s，只比理想值高约 0.6%。因此动态调度几乎消除了全部负载不均衡。

### 10.4 为什么 1x4 和 2x2 几乎打平

两者都使用 4 个线程，也都把昂贵行分散开了。在 2x2 中，每个 rank 拥有固定的 300 行带，但上下两个带各自包含大约一半的昂贵中心区域，所以两个 rank 的工作量相近。

动态调度平衡了每个 rank 内部的 2 个线程，剩下的 rank 级差异很小。0.1823 s 和 0.1833 s 之间 0.5% 的差距在运行波动范围内，所以在这个 VM 上两种配置实际上打平。

### 10.5 为什么 4x1 仍是 2.34x

4x1 使用同一个动态二进制程序，但是：

```bash
OMP_NUM_THREADS=1
```

每个 MPI rank 只有 1 个线程。`schedule(dynamic)` 没有第二个线程可以分配工作，所以调度不起作用。

4x1 的时间保持在 0.3093 s，几乎等于 static 4x1 的 0.310 s。这就是控制实验：

- 如果加速来自动态调度，4x1 不应该改善；
- 它确实没有改善；
- 因此 1x4 和 2x2 的改善来自 OpenMP 动态调度，而不是重新编译或 MPI 本身。

### 10.6 为什么加速比不是精确的 4.00x

即使最好的配置也有一小段差距：

- `MPI_Gatherv` 和最后的 barrier；
- OpenMP 线程启动和调度；
- 图像末尾残留的尾巴；
- VM 的 vCPU 与宿主机物理核共享；
- 只重复了三次，仍有测量噪声；
- 串行计时不含 PPM 写入，而并行计时包含 gather，这让并行结果略吃亏。

因此测得的 99% 效率应理解为“在这次特定的 4 vCPU 测量中接近理想”，而不是普遍结论。

## 11. 正确性检查

```bash
./scripts/run.sh
```

最后打印的 6 条 SHA-256 必须完全相同。期望的摘要是：

```text
4606e1e408496266fd928852c8b0c4137113773c7d01f4b814b3ad058827c17e
```

如果某条哈希不同，说明某一行丢失、重复，或被写到了错误的偏移位置。确定性的像素函数和 `-ffp-contract=off` 使这个检查非常严格。

## 12. 仓库结构

```text
.
├── Makefile
├── README.md
├── README.zh-CN.md
├── docs/
│   └── mandelbrot.png
├── scripts/
│   └── run.sh
├── src/
│   ├── mandelbrot_serial.c
│   ├── mandelbrot_mpi_omp.c
│   └── mandelbrot_mpi_omp_static.c
└── steps/
    ├── README.md
    ├── Makefile
    ├── hello_mpi.c
    ├── mpi_rows.c
    ├── mandelbrot_serial.c
    ├── mandelbrot_mpi.c
    ├── mandelbrot_mpi_omp_static.c
    └── mandelbrot_mpi_omp_dynamic.c
```

- `mandelbrot_serial.c`：串行参考实现。
- `mandelbrot_mpi_omp.c`：使用动态调度的最终混合版本。
- `mandelbrot_mpi_omp_static.c`：相同混合代码的静态调度版本。
- `scripts/run.sh`：运行完整测量矩阵并比较哈希。
- `steps/`：从实验最开始的逐步程序：MPI hello、行划分、串行渲染器、纯 MPI、静态混合和动态混合。逐步命令见 [`steps/README.md`](steps/README.md)。
- `README.zh-CN.md`：本文档的中文翻译。

## 13. 从零复现

```bash
git clone https://gitcode.com/JiakunPi/mandelbrot-mpi-openmp.git
cd mandelbrot-mpi-openmp
make
./scripts/run.sh
```

最终哈希应与上面的摘要完全一致。

## 14. 局限与未来工作

- 扩展性研究是单机的；两节点检查确认了进程放置和正确性，但不是完整的多节点扩展性研究。
- MPI 使用连续行带。循环行分配或动态 MPI tile 队列也可以平衡 rank，但会让汇总更复杂、消息更多。
- 图像较小（1.4 MB），所以测得的效率对大图像来说偏乐观。
- 静态测量是单次观测；动态和串行数值是三次运行的平均值。
- 可能的扩展：更大图像、AVX/SIMD 内层循环、代价感知的 tile 调度，以及混合 MPI tile 队列。

## 15. 报告

完整实验报告单独提交，不包含在这个仓库中。
