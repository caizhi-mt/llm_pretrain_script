# MATE CPU 绑核指南

本文说明如何为 MUSA MoE 训练中的 MATE 提交线程配置 CPU affinity，覆盖 Intel Xeon 与 AMD EPYC 常见拓扑。

## 1. 作用范围

绑核功能默认关闭。推荐配置为：

```bash
export MUSA_CPU_AFFINITY=1
export MUSA_CPU_AFFINITY_MODE=mate
export MUSA_CPU_AFFINITY_MAP='0-7;8-15;16-23;24-31;32-39;40-47;48-55;56-63'
```

`MUSA_CPU_AFFINITY_MAP` 使用分号分隔 local rank，每个 rank 内使用逗号和连字符表示 CPU：

```text
rank0 CPUs ; rank1 CPUs ; ... ; rankN CPUs
```

例如 `0-3,8;4-7,9` 表示 local rank 0 可运行在 CPU 0、1、2、3、8，local rank 1 可运行在 CPU 4、5、6、7、9。

推荐使用 `MUSA_CPU_AFFINITY_MODE=mate`。该模式在 DeepEP 初始化完成后，只绑定进入 MATE forward 的 Python 提交线程，不限制 DeepEP、MCCL、OpenMP 和其他通信线程。

不推荐生产训练使用 `early`：

```bash
export MUSA_CPU_AFFINITY_MODE=early
```

`early` 会在导入 PyTorch 前绑定进程，后续创建的线程会继承较窄的 cpuset。实测它会让 DeepEP ACE 通信线程受限并增加 rank 间等待。

## 2. 通用原则

Intel 和 AMD 都遵循以下原则：

1. 以 `mthreads-gmi topo -m` 报告的 GPU `NUMA Affinity` 为准，不按 GPU 编号猜测 NUMA。
2. local rank 的 CPU 集合优先来自该 GPU 所在 NUMA node。
3. 初始配置只选择每个物理核的一个逻辑 CPU，不同时选择 SMT/Hyper-Threading sibling。
4. 同一个 NUMA node 上的多个 local rank 均分物理核，集合不要重叠。
5. map 顺序对应 `LOCAL_RANK`，也就是 `MUSA_VISIBLE_DEVICES` 中的可见设备顺序。重排设备后必须同步重排 map。
6. 建议每个 rank 从 4～8 个物理核开始 A/B；更多 CPU 不一定更快。
7. affinity 只限制目标线程可以运行在哪里，不会为它独占这些 CPU。需要完全隔离时还要配合 cgroup、systemd AllowedCPUs 或 `isolcpus`。

## 3. 收集拓扑

### 3.1 查看 CPU、物理核和 NUMA

```bash
lscpu
lscpu -e=CPU,CORE,SOCKET,NODE,ONLINE
numactl --hardware
```

容器中没有 `numactl` 时，至少执行：

```bash
cat /sys/devices/system/node/node*/cpulist
grep Cpus_allowed_list /proc/self/status
```

下面的命令按 `(socket, core)` 去重，每个物理核只保留第一个逻辑 CPU：

```bash
lscpu -p=CPU,CORE,SOCKET,NODE \
  | awk -F, '!/^#/ { key=$3 ":" $2; if (!seen[key]++) print "node=" $4, "socket=" $3, "core=" $2, "cpu=" $1 }' \
  | sort -V
```

不要假设 CPU 编号的前半部分一定都是首个 SMT thread；必须用 `CORE` 和 `SOCKET` 去重确认。

### 3.2 查看 GPU NUMA

```bash
mthreads-gmi topo -m
```

重点记录每个 GPU 的两列：

- `CPU Affinity`：与 GPU 接近的全部逻辑 CPU；
- `NUMA Affinity`：GPU 所在 NUMA node。

也可以从 PCI 设备确认：

```bash
for dev in /sys/bus/pci/devices/*; do
  if [[ -r "$dev/numa_node" ]]; then
    printf '%s numa=%s\n' "${dev##*/}" "$(cat "$dev/numa_node")"
  fi
done
```

### 3.3 检查容器 cpuset

```bash
taskset -pc $$
grep Cpus_allowed_list /proc/self/status
```

`MUSA_CPU_AFFINITY_MAP` 中的 CPU 必须属于容器允许的集合。代码会校验这一点，越界时直接报错，而不是静默使用错误映射。

## 4. Intel Xeon 绑核

### 4.1 普通双路 NUMA

Intel Xeon 常见配置是每个 socket 一个 NUMA node，Hyper-Threading 提供两个逻辑 CPU/物理核。先从每个物理核选择一个 thread，再在同 NUMA node 内按 GPU 数量均分。

本项目验证机为：

```text
CPU: Intel Xeon Gold 6530
socket: 2
physical cores/socket: 32
threads/core: 2
NUMA nodes: 2
GPU0～GPU3: NUMA 0
GPU4～GPU7: NUMA 1
```

逻辑 CPU 分布：

```text
NUMA 0: 0-31,64-95
NUMA 1: 32-63,96-127
```

其中 `0-31` 和 `64-95` 是相同 32 个物理核的两个 Hyper-Threading sibling；`32-63` 和 `96-127` 同理。选择首个 thread 后，8 卡映射为：

```bash
export MUSA_CPU_AFFINITY=1
export MUSA_CPU_AFFINITY_MODE=mate
export MUSA_CPU_AFFINITY_MAP='0-7;8-15;16-23;24-31;32-39;40-47;48-55;56-63'
```

该映射保证 GPU0～GPU3 使用 NUMA 0 的物理核，GPU4～GPU7 使用 NUMA 1 的物理核，并且 local rank 之间不重叠。

### 4.2 Intel SNC

启用 Sub-NUMA Clustering 后，一个 socket 可能拆成 2 或 4 个 NUMA node。此时不要继续按 socket 平分 CPU：

1. 从 `mthreads-gmi topo -m` 读取每张 GPU 的真实 NUMA node；
2. 从 `lscpu -e` 筛选该 node 的首个 SMT thread；
3. 只在这个 node 内给对应 local rank 分配 CPU。

如果一个 GPU 的 `NUMA Affinity` 为 `2`，它的 map 应来自 node 2，而不是简单地按“前四卡属于 socket 0”处理。

## 5. AMD EPYC 绑核

AMD EPYC 的关键差异是 NPS（NUMA Per Socket）模式。常见配置包括：

- NPS1：每个 socket 1 个 NUMA node；
- NPS2：每个 socket 2 个 NUMA node；
- NPS4：每个 socket 4 个 NUMA node。

因此双路 EPYC 可能呈现 2、4 或 8 个 NUMA node。CPU 编号还可能按 CCD/SMT 交错，不能照搬 Intel 示例中的连续编号或“总 CPU 数除以二”规则。

### 5.1 AMD NPS1

NPS1 的方法与普通双路 Intel 类似：

1. 确认 GPU0～GPU3 是否位于 socket 0/node 0，GPU4～GPU7 是否位于 socket 1/node 1；
2. 对 `(socket, core)` 去重，只保留一个 SMT thread；
3. 将每个 node 的物理核平均分给本 node 的 GPU。

假设 `lscpu -e` 显示 node 0 的首线程为 `<N0_CPUS>`、node 1 为 `<N1_CPUS>`，并且每个 node 有 4 张 GPU，则将 `<N0_CPUS>` 分成 rank0～rank3 四组，将 `<N1_CPUS>` 分成 rank4～rank7 四组。不要直接使用文档中的 Intel CPU 编号。

### 5.2 AMD NPS2/NPS4

NPS2/NPS4 下应按 GPU 的 NUMA node 或 CCD 邻近性分配：

| Local rank | GPU NUMA | CPU 集合来源 |
|---|---:|---|
| 0 | `mthreads-gmi` 的 GPU0 NUMA | 该 node 的首 SMT thread |
| 1 | GPU1 NUMA | 该 node 的首 SMT thread |
| ... | ... | ... |
| 7 | GPU7 NUMA | 该 node 的首 SMT thread |

若每张 GPU 对应一个独立 NUMA node，可直接为每个 rank 选择该 node 内的 4～8 个物理核。若多张 GPU 共享一个 node，则均分该 node 的物理核。

EPYC 的一个 CCD 通常形成独立 LLC 域。CPU 数量足够时，优先让一个 rank 的集合落在同一个 CCD/LLC 域内；跨 CCD 但不跨 NUMA 是次选，跨 NUMA 应避免。可以用以下命令辅助确认 cache 域：

```bash
lscpu -e=CPU,CORE,SOCKET,NODE,CACHE
```

### 5.3 AMD SMT 与性能模式

AMD 同样建议先只选择每个物理核的一个 SMT thread。如果 MATE 提交线程仍频繁被抢占，再测试加入 sibling，不能直接假定双 SMT 更快。

可选地检查 CPU governor：

```bash
grep . /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor | sort -u
```

有 root 权限且平台支持时：

```bash
cpupower frequency-set -g performance
```

使用 `amd-pstate-epp` 的系统还应检查：

```bash
grep . /sys/devices/system/cpu/cpu*/cpufreq/energy_performance_preference | sort -u
```

BIOS、内核或集群策略不允许修改时，不要强制写 sysfs；记录当前模式并保持 A/B 两侧一致。

## 6. 启动和回退

这些变量必须在启动 Python/torchrun 前设置：

```bash
export MUSA_CPU_AFFINITY=1
export MUSA_CPU_AFFINITY_MODE=mate
export MUSA_CPU_AFFINITY_MAP='...每个 local rank 的 CPU 集合...'
```

关闭绑核：

```bash
export MUSA_CPU_AFFINITY=0
```

`llm_pretrain_script/cluster/dist_run_megatron.sh` 会安全透传上述三个变量，包括含分号的 map。同构节点可以直接复用同一 map。多机训练若存在不同 CPU/GPU 拓扑，必须在每个节点的启动包装脚本中分别生成并设置 map，不应通过当前 launcher 把一台机器的 CPU 编号复制到全部节点。

## 7. 验证是否生效

每个 local rank 应打印一次：

```text
[MUSA_CPU_AFFINITY] mode=mate local_rank=0 native_thread_id=12345 cpus=[0, 1, ...]
```

从日志检查：

```bash
grep -R '\[MUSA_CPU_AFFINITY\]' output_log/
```

使用日志中的 `native_thread_id` 检查真实 affinity：

```bash
taskset -pc <native_thread_id>
grep Cpus_allowed_list /proc/<pid>/task/<native_thread_id>/status
```

同时确认 DeepEP 线程没有继承窄 affinity：

```bash
ps -L -p <pid> -o pid,tid,psr,comm
```

验收时至少进行一次关闭/开启 A/B：

1. 固定代码、batch、sequence length、并行拓扑、dtype 和数据；
2. 分别设置 `MUSA_CPU_AFFINITY=0/1`；
3. 排除首个 warmup step；
4. 比较至少 10 个稳态 step 的平均值、中位数、最小值和最大值；
5. Trace 中同时检查 DeepEP ACE 时间、permute→GroupGEMM 空泡以及 GroupGEMM kernel 时间。

## 8. 常见问题

### 请求的 CPU 不在容器 cpuset

错误示例：

```text
requested CPUs outside its cpuset
```

先扩大容器的 `--cpuset-cpus`/cgroup 配置，或者只从 `Cpus_allowed_list` 中构造 map。

### map 项数少于 local rank 数

每个 local rank 都必须有非空项。例如单机 8 卡需要至少 8 个分号分隔的集合。

### DeepEP 变慢

确认使用的是：

```bash
export MUSA_CPU_AFFINITY_MODE=mate
```

如果使用 `early`，DeepEP/通信线程可能继承窄 cpuset。切回 `mate` 后重新采集 trace。

### 绑核后没有收益

绑核主要降低 Python MATE 提交线程的调度长尾，不保证每台机器都有收益。若 CPU 没有争用或 GPU 关键路径由计算/通信主导，端到端变化可能低于测量噪声，应保留 `MUSA_CPU_AFFINITY=0` 的回退路径。
