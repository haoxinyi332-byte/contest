import global_state
from global_state import *
def select_victim(wafers, state):
    """基于多因素代价模型选择牺牲晶圆"""
    min_cost = float('inf')
    selected = None
    current_time = get_now(state.start_time)

    for wid in wafers:
        # 1. 计算已加工时间代价
        processed_time = calculate_processed_time(wid, state)
        time_cost = processed_time * GlobalConfig.SACRIFICE_WEIGHTS['processed_time']

        # 2. 计算剩余步骤代价
        remaining_steps = estimate_remaining_steps(wid, state)
        steps_cost = remaining_steps * GlobalConfig.SACRIFICE_WEIGHTS['remaining_steps']

        # 3. 计算资源优先级代价
        resource_priority = get_resource_priority(wid, state)
        priority_cost = (1 - resource_priority) * GlobalConfig.SACRIFICE_WEIGHTS['resource_priority']

        # 4. 计算JIT紧急度代价
        jit_cost =(1 - calculate_jit_urgency(wid, current_time, state)) * GlobalConfig.SACRIFICE_WEIGHTS['jit_urgency']

        # 总代价 = 各因素加权和
        total_cost = time_cost + steps_cost + priority_cost + jit_cost

        # 选择最小代价的晶圆
        if total_cost < min_cost:
            min_cost = total_cost
            selected = wid

    # 记录调试信息
    with state.mutex_print:
        if selected:
            print(f"Selected victim wafer {selected} with cost {min_cost:.2f}")
    return selected
def estimate_remaining_steps(wid, state):
    """更精确的剩余步骤估算"""
    current_module, _ = state.wafer_current_location.get(wid, (None, None))
    template = GlobalConfig.path_templates["B"]
    try:
        current_idx = next(i for i, x in enumerate(template)
                           if x == current_module or
                           (isinstance(x, list) and current_module in x))
        return len(template) - current_idx - 1
    except StopIteration:
        return 5  # 默认值
def rollback_wafer(wid, state, excluded_wafers=None):
    """回滚晶圆到上一个安全点，支持LL槽选择和牺牲替换逻辑"""
    if excluded_wafers is None:
        excluded_wafers = set()

    current_module, _ = state.wafer_current_location.get(wid, (None, None))

    # 1. 从当前PM移除
    if current_module and current_module.startswith("PM"):
        status = state.pm_status[current_module]
        with status["lock"]:  # 避免并发问题
            if status["current_wafer"] == wid:
                status.update({
                    "in_use": False,
                    "current_wafer": None,
                    "idle_start": get_now(state.start_time)
                })
                with state.mutex_print:
                    print(f"⚠️ 强制终止晶圆{wid}在{current_module}的加工")

    # 2. 找到最近经过的LL
    last_ll = find_last_ll(wid, state)
    if not last_ll:
        return False

    # 3. 回滚到对应LL槽位（LLC/D用S1；LLA/B用S2）
    with state.ll_slot_lock[last_ll]:
        if last_ll in ["LLC", "LLD"]:
            if state.ll_slots[last_ll]["S1"] is None:
                state.ll_slots[last_ll]["S1"] = wid
                state.wafer_current_location[wid] = (last_ll, get_now(state.start_time))
                with state.mutex_print:
                    print(f"🔄 晶圆{wid}回滚到 {last_ll}.S1")
                return True
        elif last_ll in ["LLA", "LLB"]:
            if state.ll_slots[last_ll]["S2"] is None:
                state.ll_slots[last_ll]["S2"] = wid
                state.wafer_current_location[wid] = (last_ll, get_now(state.start_time))
                with state.mutex_print:
                    print(f"🔄 晶圆{wid}回滚到 {last_ll}.S2")
                return True
            else:
                # 如果S2满了，把原晶圆退回LP
                occupied_wid = state.ll_slots[last_ll]["S2"]
                lp_module = get_original_lp(occupied_wid, state)
                if lp_module:
                    state.ll_slots[last_ll]["S2"] = wid  # 把当前回滚晶圆塞进去
                    state.wafer_current_location[wid] = (last_ll, get_now(state.start_time))
                    state.wafer_current_location[occupied_wid] = (lp_module, get_now(state.start_time))
                    # with state.mutex_print:
                    #     print(f"🔄 晶圆{wid}回滚到 {last_ll}.S2")
                    #     print(f"↩️ 晶圆{occupied_wid} 被移回原始LP：{lp_module}")
                    return True

    # 4. 如果失败，尝试换另一个代价更低的晶圆回滚
    excluded_wafers.add(wid)
    deadlock_cycle = identify_deadlock_cycle(state)
    remaining = [w for w in deadlock_cycle if w not in excluded_wafers]

    if not remaining:
        return False  # 无可用晶圆可替代

    new_victim = select_victim(remaining, state)
    if new_victim:
        return rollback_wafer(new_victim, state, excluded_wafers)

    return False
def resolve_deadlock(state):
    """增强的死锁解决方案"""
    deadlock_wafers = identify_deadlock_cycle(state)
    if not deadlock_wafers:
        return False

    victim = select_victim(deadlock_wafers, state)
    if not victim:
        return False

    # 执行回滚前记录原因
    reason = get_sacrifice_reason(victim, state)

    if rollback_wafer(victim, state):
        state.deadlock_count += 1
        state.sacrifice_stats['total'] += 1
        state.sacrifice_stats['by_reason'][reason] += 1

        with state.mutex_print:
            state.error_log.append((
                victim,
                f"Deadlock resolved by rollback. Reason: {reason}"
            ))
        return True
    return False

def run_simulation_b1():
    """主运行函数"""
    # 初始化系统状态
    state = SystemState()
    state.start_time = time.time()
    # 初始化调度器（会同时初始化带可视化的ACO）
    scheduler = HierarchicalScheduler(state)
    plot_counter = 0
    last_plot_time = time.time()
    # 创建绘图窗口（提前初始化避免线程冲突）
    if GlobalConfig.PLOT_ENABLED:
        plt.ion()
        fig, axs = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle('实时调度监控')
    # 初始化调度器
    # scheduler = HierarchicalScheduler(state)

    # 添加多目标优化记录
    history_total_time = []
    history_throughput = []
    history_jit_violation = []

    try:
        # ================= 1. 创建晶圆任务 =================
        wafer_tasks = []
        wafer_id = 1
        # 任务一(b)配置：所有LP都使用路径B，按晶圆盒顺序处理
        for lp_index in range(1, GlobalConfig.LP_COUNT + 1):
            # 创建当前晶圆盒的任务
            for wafer_in_lp in range(1, GlobalConfig.LP_WAFER_COUNT + 1):
                matid = f"{lp_index}.{wafer_in_lp}"
                lp_name = f"LP{lp_index}"
                path_type = "B"  # 所有晶圆盒使用路径B
                wafer_tasks.append((wafer_id, matid, lp_name, path_type))
                wafer_id += 1
                time.sleep(0.1 * GlobalConfig.TIME_SCALE)
        # ================= 2. 启动晶圆处理线程 =================
        state.wafer_tasks = wafer_tasks
        with ThreadPoolExecutor(max_workers=GlobalConfig.WAFER_WORKERS) as executor:
            futures = [
                executor.submit(wafer_process, wid, matid, lpname, path_type, state, scheduler)
                for (wid, matid, lpname, path_type) in wafer_tasks
            ]
            # ================= 3. 主监控循环 =================
            while any(not f.done() for f in futures):
                now = get_now(state.start_time)
                # 执行局部调整
                scheduler.local_adjust()
                if (GlobalConfig.PLOT_ENABLED
                        and len(scheduler.aco.history['time']) > 1  # 确保有数据
                        and (time.time() - last_plot_time > GlobalConfig.PLOT_TIME_INTERVAL
                             or plot_counter % GlobalConfig.PLOT_UPDATE_INTERVAL == 0)):
                    global_state._update_real_time_plots(axs, state, scheduler.aco)
                    last_plot_time = time.time()

                # 定期检查死锁
                if now - state.last_deadlock_check > state.deadlock_check_interval:
                    state.last_deadlock_check = now
                    if detect_deadlock(state):
                        print("\n⚠️ 检测到死锁，尝试解决...")
                        if resolve_deadlock(state):
                            print("✅ 死锁已解决")
                        else:
                            print("❌ 死锁解决失败")
                    print_progress(state)

                # 记录多目标优化数据
                if state.wafer_last_leave_time:
                    current_total_time = max(state.wafer_last_leave_time.values())
                    current_throughput = len(
                        [w for w in state.wafer_last_leave_time if state.wafer_last_leave_time[w] > 0]) / (
                                             current_total_time if current_total_time > 0 else 1)
                    current_jit_violation = len(state.jit_violations)

                    history_total_time.append(current_total_time)
                    history_throughput.append(current_throughput)
                    history_jit_violation.append(current_jit_violation)

                # 显示进度
                completed = len([w for w in state.wafer_last_leave_time
                                 if state.wafer_last_leave_time[w] > 0])
                alive = sum(1 for f in futures if not f.done())
                print(f"\r进度: {completed}/{len(wafer_tasks)} 晶圆 | 活跃线程: {alive}", end="")
                time.sleep(1)
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断仿真...")
    except Exception as e:
        with state.mutex_print:
            state.error_log.append(("Main", f"仿真异常: {str(e)}"))
    finally:
        if GlobalConfig.PLOT_ENABLED:
            # 保存ACO参数进化图
            scheduler.aco.save_adaptation_report("aco_parameters_1b.png")
            # 保存实时监控图
            fig.savefig("real_time_monitor_1b.png", dpi=300)
            plt.close(fig)
        # ================= 4. 清理资源 =================
        print("\n正在停止所有线程...")
        state.scheduler_active = False
        # ================= 5. 输出结果 =================
        print("\n 生成最终报告...")

        with open("b1_json.json", "w") as f:
            json.dump({"MoveList": state.move_list}, f, indent=4)
        # # 绘制甘特图
        draw_gantt(state.gantt_data,"b")

        # 绘制多目标优化趋势图
        plot_optimization_trend(history_total_time, history_throughput, history_jit_violation)

        # 输出统计
        if state.wafer_last_leave_time:
            total_time = max(state.wafer_last_leave_time.values())
            print(f"\n 总处理时间: {total_time/GlobalConfig.TIME_SCALE:.2f}秒")
        print(f" JIT违规次数: {len(state.jit_violations)}")