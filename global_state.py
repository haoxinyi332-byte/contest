import threading
import time
import json
import random
import numpy as np
from threading import Thread, Semaphore, Lock
from collections import deque, defaultdict
import matplotlib.pyplot as plt
from matplotlib import rcParams
from concurrent.futures import ThreadPoolExecutor

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False


# ================= 全局配置 =================
class GlobalConfig:
    # 工艺处理时间（秒）
    PROCESS_TIMES = {
        "PM7": 70, "PM8": 70, "PM9": 200, "PM10": 200,
        "PM1": 600, "PM2": 600, "PM3": 600, "PM4": 600,"PM5": 600, "PM6": 600
    }

    # 清洁条件及时间
    CLEAN_CONDITIONS = {
        "idle": {"threshold": 80, "time": 30},
        "path_change": {"time": 200},
        "wafer_count": {"threshold": 13, "time": 300}
    }

    # 各种时间参数
    PUMP_TIME = 15  # 抽真空时间
    VENT_TIME = 20  # 充气时间
    OPEN_DOOR_TIME = 1  # 开门时间
    CLOSE_DOOR_TIME = 1  # 关门时间
    PICK_TIME = 4  # 取晶圆时间
    PLACE_TIME = 4  # 放晶圆时间
    TRANSFER_TIME_TM1 = 1  # TM1移动时间
    ALIGN_TIME = 8  # 校准时间
    COOL_TIME = 70  # 冷却时间

    # JIT驻留时间限制
    JIT_PM_STAY_LIMIT = 15  # PM内驻留时间限制
    JIT_TRANSFER_LIMIT = 30  # 转移时间限制

    # LP数量和每盒晶圆数量
    LP_COUNT = 3
    LP_WAFER_COUNT = 25

    # TIME缩放比例（加快仿真）
    TIME_SCALE = 0.001

    # 初始化信息素表
    pheromone_table = {
        ("PM7", "PM8"): {"PM7": 1.0, "PM8": 1.0},
        ("PM9", "PM10"): {"PM9": 1.0, "PM10": 1.0},
        ("PM1", "PM2", "PM3", "PM4", "PM5", "PM6"): {"PM1": 1.0, "PM2": 1.0, "PM3": 1.0, "PM4": 1.0, "PM5": 1.0, "PM6": 1.0}
    }

    # 信息素挥发率
    pheromone_evaporation = 0.1

    # 信息素奖励系数
    pheromone_boost = 1.5
    PLOT_ENABLED = True  # 全局绘图开关
    PLOT_UPDATE_INTERVAL = 10  # 绘图更新间隔（次）
    PLOT_TIME_INTERVAL = 5.0  # 绘图更新间隔（秒）
    # 机械手臂对应关系
    OPPOSITE_ARMS = {
        "TM2": {
            "LLA": "LLD", "LLD": "LLA",
            "PM6": "PM8", "PM8": "PM6",
            "PM7": "PM10", "PM10": "PM7",
            "LLC": "LLB", "LLB": "LLC"
        },
        "TM3": {
            "LLC": "PM4", "PM4": "LLC",
            "PM1": "PM5", "PM5": "PM1",
            "PM2": "PM6", "PM6": "PM2",
            "PM3": "LLD", "LLD": "PM3"
        }
    }
    # 牺牲函数权重配置
    SACRIFICE_WEIGHTS = {
        'processed_time': 0.4,  # 已加工时间权重
        'remaining_steps': 0.2,  # 剩余步骤权重
        'resource_priority': 0.3,  # 资源优先级权重
        'jit_urgency': 0.5  # JIT紧急度权重
    }
    # 资源优先级定义 (值越高表示资源越关键)
    RESOURCE_PRIORITY = {
        "PM1": 0.9, "PM2": 0.9, "PM3": 0.9, "PM4": 0.9, "PM5": 0.9, "PM6": 0.9,  # 关键工艺模块
        "PM7": 1.0, "PM8": 1.0, "PM9": 1.0, "PM10": 1.0,  # 次要工艺模块
        "TM2": 0.8, "TM3": 0.8,  # 传输机械臂
        "LLA": 0.6, "LLB": 0.6, "LLC": 0.7, "LLD": 0.7  # 装载锁
    }    # TM2和TM3的环形路径
    tm2_ring = ["LLA", "PM9", "PM7", "LLC", "LLD", "PM8", "PM10", "LLB"]
    tm3_ring = ["LLC", "PM1", "PM2", "PM3", "PM4", "PM5", "PM6", "LLD"]
    path_templates = {
        "A": ["LLA"] + ["PM7", "PM8", "LLC", "LLD2", "LLB"] * 5,
        "B": [["LLA", "LLB"], ["PM7", "PM8"], "LLC", ["PM1", "PM2"], "LLD2", "LLA"],
        "C": [["LLA", "LLB"], ["PM7", "PM8"], "LLC", ["PM1", "PM2", "PM3", "PM4"], "LLD", ["PM9", "PM10"],"LLA"],
        "D": [["LLA", "LLB"], ["PM7", "PM8"], ["PM9", "PM10"], "LLD2", "LLA"],
        "E": [["LLA", "LLB"], ["PM7", "PM8"], "LLC", "PM1", "LLD2", "LLA"],
        "F": [["LLA", "LLB"], ["PM7", "PM8"], "LLC", "PM2", "LLD2", "LLA"],
        "G": [["LLA", "LLB"], ["PM7", "PM8"], "LLC", "PM3", "LLD2", "LLA"],
        "H": [["LLA", "LLB"], ["PM7", "PM8"], "LLC", "PM4", "LLD2", "LLA"],
        "I": [["LLA", "LLB"], ["PM7", "PM8"], "PM9", "LLD2", "LLA"],
        "J": [["LLA", "LLB"], ["PM7", "PM8"], "PM10", "LLD2", "LLA"],
        "K": [["LLA", "LLB"], ["PM7", "PM8"], "LLC", ["PM1", "PM2", "PM3", "PM4"], "LLD", ["PM9", "PM10"],"LLA"]
    }
    WAFER_WORKERS = 75
# ================= 系统状态 =================
class SystemState:
    def __init__(self):
        # 信号量
        self.sem_TM1 = Semaphore(1)
        self.sem_TM2 = Semaphore(1)
        self.sem_TM3 = Semaphore(1)
        self.sem_AL = Semaphore(1)
        self.sem_PM = {f"PM{i}": Semaphore(1) for i in range(1, 11)}
        self.sem_LL = {name: Semaphore(2 if name in ["LLA", "LLB"] else 1) for name in ["LLA", "LLB", "LLC", "LLD"]}
        self.sem_LP = {f"LP{i}": Semaphore(1) for i in range(1, GlobalConfig.LP_COUNT + 1)}
        self.move_type_dict = {
            "PickMove": 1,
            "PlaceMove": 2,
            "TransferMove": 3,
            "PrepareMove": 4,
            "CompleteMove": 5,
            "PumpMove": 6,
            "VentMove": 7,
            "ProcessMove": 8,
            "CleanMove": 9,
            "AlignMove": 10,
        }
        # LL模块状态
        self.ll_state = {"LLA": "atm", "LLB": "atm", "LLC": "atm", "LLD": "atm"}
        self.ll_slots = {name: {"S2": None, "S1": None} for name in ["LLA", "LLB", "LLC", "LLD"]}
        self.ll_slot_lock = {name: Lock() for name in ["LLA", "LLB", "LLC", "LLD"]}
        self.ll_state_lock = {name: Lock() for name in ["LLA", "LLB", "LLC", "LLD"]}

        # 互斥锁
        self.mutex_print = Lock()
        self.move_list_lock = Lock()
        self.gantt_data_lock = Lock()
        self.error_log = []  # 异常日志
        # 添加死锁检测相关状态
        self.deadlock_check_interval = 5.0  # 死锁检测间隔(秒)
        self.last_deadlock_check = 0.0
        self.deadlock_count = 0
        self.deadlock_resolution_enabled = True
        # PM模块状态
        self.pm_status = {f"PM{i}": {
            "last_end_time": 0.0,
            "last_path": None,
            "count": 0,
            "idle_start": 0.0,
            "cleaning": False,
            "in_use": False,
            "current_wafer": None,
            "start_time": 0.0
        } for i in range(1, 11)}
        # self.pm_queue = {f"PM{i}": deque() for i in range(1, 11)}
        # self.pm_queue_lock = {f"PM{i}": Lock() for i in range(1, 11)}

        # JIT跟踪
        self.wafer_last_leave_time = {}
        self.wafer_current_location = {}
        self.jit_violations = set()

        # 移动ID计数器
        self.move_id_counter = [0]

        # 调度器状态
        self.scheduler_active = True
        self.reschedule_thread = None
        self.start_time = None
        self.move_list = []
        self.gantt_data = []
        # 添加晶圆任务记录
        self.wafer_tasks = []  # 格式: [(wid, matid, lpname, path_type), ...]
        # 牺牲统计
        self.sacrifice_stats = {
            'total': 0,
            'by_reason': defaultdict(int)
        }
        self.last_thread_count = 0
        self.thread_count_unchanged_since = time.time()
        self.semaphore_status_interval = 100  # 30秒无变化时打印信号量状态
        self.CHECK_JIT_COUNT = 0
        self.last_clean_check_time = 0
        self.clean_check_interval = 5  # 每5秒检查一次
        self.lp_next_expected = {f"LP{i}": 1 for i in range(1, GlobalConfig.LP_COUNT + 1)}  # 下一个应出片的编号
        self.lp_sequence_lock = {f"LP{i}": Lock() for i in range(1, GlobalConfig.LP_COUNT + 1)}
        self.lp_sequence_cond = {f"LP{i}": threading.Condition(self.lp_sequence_lock[f"LP{i}"])
                                 for i in range(1, GlobalConfig.LP_COUNT + 1)}
def terminate_all_threads(state, timeout=5.0):
    """安全终止所有活跃线程"""
    state.scheduler_active = False
    # 第二步：强制释放关键资源
    with state.thread_lock:
        for tid, (thread, wafer_id) in list(state.active_threads.items()):
            try:
                # 释放可能持有的信号量
                for sem in [state.sem_TM1, state.sem_TM2, state.sem_TM3]:
                    if sem._value == 0:  # 如果被占用
                        sem.release()

                # # 从PM队列移除
                # for pm in state.pm_queue:
                #     if wafer_id in state.pm_queue[pm]:
                #         state.pm_queue[pm].remove(wafer_id)

                # 从LL槽位移除
                for ll in state.ll_slots:
                    for slot in ["S1", "S2"]:
                        if state.ll_slots[ll][slot] == wafer_id:
                            state.ll_slots[ll][slot] = None
            except Exception as e:
                print(f"清理晶圆{wafer_id}资源时出错: {e}")

    # 第三步：等待线程自然结束
    start_time = time.time()
    while state.active_threads and (time.time() - start_time) < timeout:
        time.sleep(0.1)

    # 第四步：强制终止残留线程
    if state.active_threads:
        with state.thread_lock:
            for tid, (thread, _) in list(state.active_threads.items()):
                try:
                    thread._stop()  # 注意：可能有副作用
                    print(f"强制终止线程{tid}")
                except:
                    pass

    # 最终状态检查
    alive = sum(1 for t in threading.enumerate() if t != threading.main_thread())
    print(f"清理完成，剩余活跃线程: {alive}")

def emergency_recovery(state):
    with state.mutex_print:
        print("\n系统长时间未反应，出现可能死锁 ⚠️触发紧急恢复流程...")
    # 初始化统计数据
    recovered_wafers = []
    remaining_steps = defaultdict(int)
    max_additional_time = 0
    current_time = get_now(state.start_time)

    # 获取所有需要恢复的晶圆
    wafers_to_recover = [
        (wid, loc[0]) for wid, loc in state.wafer_current_location.items()
        if not loc[0].startswith("LP")
    ]
    # 预计算各模块的标准处理时间
    module_times = {
        **{f"PM{i}": GlobalConfig.PROCESS_TIMES[f"PM{i}"] for i in range(1, 11)},
        "AL": GlobalConfig.ALIGN_TIME,
        "LL_transfer": max(GlobalConfig.PUMP_TIME, GlobalConfig.VENT_TIME)+GlobalConfig.PLACE_TIME,
        "TM1_transfer": GlobalConfig.TRANSFER_TIME_TM1,
        "cooling": GlobalConfig.COOL_TIME,
        "open_close":GlobalConfig.OPEN_DOOR_TIME*2 +GlobalConfig.PICK_TIME
    }
    for wid, current_loc in wafers_to_recover:
        # 获取晶圆信息
        lp_name, matid, path_type = None, None, None
        for task in state.wafer_tasks:
            if task[0] == wid:
                lp_name, matid, path_type = task[2], task[1], task[3]
                break

        if not lp_name:
            print(f"警告：晶圆{wid}未找到原始LP，跳过")
            continue

        # ===== 基于剩余路径估算时间 =====
        path_template = GlobalConfig.path_templates[path_type]
        remaining_time = 0

        # 找到当前模块在路径中的位置
        count = 0
        current_idx = 0
        for p in path_template:
            if isinstance(p, list) and current_loc in p:
                current_idx = path_template.index(p)
                break
            else:
                if p == current_loc:
                    current_idx = path_template.index(current_loc)
                    break
        count = current_idx
        remaining_path = path_template[current_idx + 1:]
        # 计算剩余路径时间
        for current_loc2 in remaining_path:
            count += 1
            if isinstance(current_loc2,list):
                current_loc2 = random.choice(current_loc2)
            if current_loc2.startswith("PM"):
                pm_time_left = module_times[current_loc2] + module_times["open_close"]
                remaining_time += pm_time_left
                remaining_steps["PM_exit"] += 1
            elif current_loc2 in ["LLC", "LLD", "LLD2"]:
                remaining_time += module_times["open_close"]
                if current_loc2 == "LLD2":
                    remaining_time += module_times["cooling"]
                remaining_steps["recovery"] += 1
            elif current_loc2 in ["LLA", "LLB"]:
                remaining_time += module_times["open_close"]
                remaining_time += module_times["LL_transfer"]
                remaining_steps["LL_recovery"] += 1
            # ===== 实际恢复操作 =====
            if current_loc2 in ["LLA", "LLB"]:
                with state.sem_TM1:
                    log_move("PickMove", "TM1", GlobalConfig.PICK_TIME, state, matid)
                    log_move("TransferMove", "TM1", GlobalConfig.TRANSFER_TIME_TM1, state, matid)
                    log_move("PlaceMove", "TM1", GlobalConfig.PLACE_TIME, state, matid)
                with state.sem_LP[lp_name]:
                    log_move("CompleteMove", lp_name, GlobalConfig.CLOSE_DOOR_TIME, state, matid)
            elif current_loc2 in ["LLC", "LLD", "LLD2"] + [f"PM{i}" for i in range(1, 11)]:
                # 从当前位置取出
                cur = current_loc2
                if current_loc2 == "LLD2":
                    current_loc2 = "LLD"
                with state.sem_TM2 if current_loc2 not in ["LLC"]+[f"PM{i}" for i in range(1, 7)] else state.sem_TM3:
                    pick_time = GlobalConfig.PICK_TIME if current_loc2 not in ["LLC"]+[f"PM{i}" for i in range(1, 7)] else GlobalConfig.PICK_TIME
                    log_move("PickMove", current_loc2, pick_time, state, matid)
                    if count == len(path_template): break
                    ll_exit = path_template[count+1]
                    if isinstance(ll_exit,str) and ll_exit == "LLD2":
                        ll_exit = "LLD"
                    if isinstance(ll_exit,list):
                        if ll_exit in ["LLA","LLB"]:
                            ll_exit = choose_LL(state)
                        elif ll_exit[0].startswith("PM"):
                            ll_exit = random.choice(ll_exit)
                    if current_loc2 in ["LLC", "LLD"]+[f"PM{i}" for i in range(1, 7)] and ll_exit in ["LLC", "LLD"]+[f"PM{i}" for i in range(1, 7)]:
                        move_time = tm3_move_time(current_loc2, ll_exit)
                        log_move("TransferMove", "TM3", move_time, state, matid)
                        log_move("PrepareMove",ll_exit,GlobalConfig.OPEN_DOOR_TIME,state,matid,slot=2)
                        log_move("PlaceMove", "TM3", GlobalConfig.PLACE_TIME, state, matid, slot=2)
                    elif current_loc2 in ["LLC", "LLD", "LLA", "LLB"]+[f"PM{i}" for i in range(7, 11)] and ll_exit in ["LLC", "LLD", "LLA", "LLB"]+[f"PM{i}" for i in range(7, 11)]:
                        move_time = tm2_move_time(current_loc2, ll_exit)
                        log_move("TransferMove", "TM2", move_time, state, matid)
                        log_move("PrepareMove",ll_exit,GlobalConfig.OPEN_DOOR_TIME,state,matid,slot=2)
                        log_move("PlaceMove", "TM2", GlobalConfig.PLACE_TIME, state, matid, slot=2)
                    if ll_exit in [f"PM{i}" for i in range(1, 11)]:
                        log_move("ProcessMove", ll_exit, GlobalConfig.PROCESS_TIMES[ll_exit],state, matid)
                    log_move("CompleteMove", ll_exit, GlobalConfig.CLOSE_DOOR_TIME, state, matid,slot=2)
        # 再用TM1运回LP
        with state.sem_TM1:
            log_move("PickMove", "TM1", GlobalConfig.PICK_TIME, state, matid)
            log_move("TransferMove", "TM1", GlobalConfig.TRANSFER_TIME_TM1, state, matid)
            log_move("PlaceMove", "TM1", GlobalConfig.PLACE_TIME, state, matid)

        with state.sem_LP[lp_name]:
            log_move("CompleteMove", lp_name, GlobalConfig.CLOSE_DOOR_TIME, state, matid)
        # 添加运输回LP的时间
        remaining_time += GlobalConfig.PICK_TIME + GlobalConfig.PLACE_TIME
        max_additional_time += remaining_time
        update_wafer_location(wid, lp_name, get_now(state.start_time), state)
        recovered_wafers.append(wid)
    # ===== 计算最坏情况总时间 =====
    total_estimated_time = current_time + max_additional_time
    throughput_rate = len(recovered_wafers) / max_additional_time if max_additional_time > 0 else 0
    with state.mutex_print:
        print(f"\n📊 恢复统计:")
        print(f"- 已回收晶圆: {len(recovered_wafers)}/{len(wafers_to_recover)}")
        print(f"- 剩余步骤分布: {dict(remaining_steps)}")
        print(f"- 最大剩余时间: {max_additional_time:.2f}s")
        print(f"- 最坏情况完成时间: {total_estimated_time:.2f}s")
        print(f"- 预估吞吐率: {throughput_rate:.2f} wafers/s")
    terminate_all_threads(state)
    return total_estimated_time
def print_progress(state):
    """增强版进度打印，包含停滞检测"""
    current_threads = threading.active_count() - 1  # 减去主线程
    now = time.time()

    # 检测线程数是否变化
    if current_threads == state.last_thread_count:
        if now - state.thread_count_unchanged_since > state.semaphore_status_interval:
            # print("\n⚠️ 系统可能停滞，当前信号量状态：")
            # for name, value in state.get_semaphore_status().items():
            #     print(f"{name}: {'锁定' if value == 0 else '空闲'}")
            # print(f"晶圆位置快照：{dict(list(state.wafer_current_location.items())[:75])}...")
            emergency_recovery(state)
            state.thread_count_unchanged_since = now  # 重置计时
    else:
        state.last_thread_count = current_threads
        state.thread_count_unchanged_since = now

    # 正常进度打印
    print(f"\r进度: {len(state.wafer_last_leave_time)}/{len(state.wafer_tasks)} 晶圆 | "
          f"活跃线程: {current_threads} | 死锁次数: {state.deadlock_count}", end="")
def detect_deadlock(state):
    """基于资源分配图的死锁检测"""
    # 构建等待图
    graph = defaultdict(set)
    # 1. 收集所有被占用的资源
    resources = {}

    for pm, status in state.pm_status.items():
        if not status["cleaning"] and status["in_use"]:
            # 当前正在PM中加工的晶圆视为占用资源
            resources[f"PM_{pm}"] = status["current_wafer"]

    # LL资源
    for ll in ["LLA", "LLB", "LLC", "LLD"]:
        with state.ll_slot_lock[ll]:
            if state.ll_slots[ll]["S1"]:
                resources[f"LL_{ll}_S1"] = state.ll_slots[ll]["S1"]
            if state.ll_slots[ll]["S2"]:
                resources[f"LL_{ll}_S2"] = state.ll_slots[ll]["S2"]

    # TM资源
    resources["TM1"] = None  # 由信号量控制
    resources["TM2"] = None
    resources["TM3"] = None

    # 2. 构建等待边
    for wafer_id, (module, _) in list(state.wafer_current_location.items()):
        next_step = get_next_step_for_wafer(wafer_id, state)  # 获取晶圆下一步需要的资源

        if next_step in resources:
            holder = resources[next_step]
            if holder != wafer_id:  # 排除自己等待自己的情况
                graph[wafer_id].add(holder)

    # 3. 检测环路
    return has_cycle(graph)
def get_next_step_for_wafer(wafer_id, state):
    """获取晶圆下一步所需的具体资源名称（用于死锁检测）"""
    current_module, _ = state.wafer_current_location.get(wafer_id, (None, None))
    if not current_module:
        return None

    # 晶圆路径查找（从move_list中找当前模块，提取下一模块）
    # 找出对应的 matid
    matid = next((m for w, m, _, _ in state.wafer_tasks if w == wafer_id), None)
    if not matid:
        return None

    # 查找 wafer 的全部 move 记录
    wafer_moves = [m for m in state.move_list if m["MatID"] == matid]
    for i, move in enumerate(wafer_moves):
        if move["ModuleName"] == current_module and i + 1 < len(wafer_moves):
            next_mod = wafer_moves[i + 1]["ModuleName"]
            if next_mod.startswith("PM"):
                return f"PM_{next_mod}"
            elif next_mod.startswith("LL"):
                # 这里假设优先占S2槽
                return f"LL_{next_mod}_S2"
            elif next_mod.startswith("LP") or next_mod == "AL":
                return "TM1"
            elif next_mod.startswith("TM"):
                return next_mod
    return None
def has_cycle(graph):
    """检测图中是否存在环路"""
    visited = set()
    recursion_stack = set()

    def dfs(node):
        visited.add(node)
        recursion_stack.add(node)

        for neighbor in graph.get(node, set()):
            if neighbor not in visited:
                if dfs(neighbor):
                    return True
            elif neighbor in recursion_stack:
                return True

        recursion_stack.remove(node)
        return False

    for node in graph:
        if node not in visited:
            if dfs(node):
                return True
    return False
def identify_deadlock_cycle(state):
    """识别死锁环中的所有晶圆"""
    graph = defaultdict(set)
    resources = {}
    for ll in ["LLA", "LLB", "LLC", "LLD"]:
        with state.ll_slot_lock[ll]:
            if state.ll_slots[ll]["S1"]:
                resources[f"LL_{ll}_S1"] = state.ll_slots[ll]["S1"]
            if state.ll_slots[ll]["S2"]:
                resources[f"LL_{ll}_S2"] = state.ll_slots[ll]["S2"]
    # 添加TM资源（由信号量控制）
    resources["TM1"] = None
    resources["TM2"] = None
    resources["TM3"] = None

    # 构建等待图
    for wafer_id, (module, _) in list(state.wafer_current_location.items()):
        next_step = get_next_step_for_wafer(wafer_id, state)
        if next_step in resources:
            holder = resources[next_step]
            if holder != wafer_id:
                graph[wafer_id].add(holder)

    # 查找环路
    cycles = []
    visited = set()

    def find_cycles(node, path):
        visited.add(node)
        path.append(node)

        for neighbor in graph.get(node, set()):
            if neighbor in path:
                cycle = path[path.index(neighbor):]
                cycles.append(cycle)
            elif neighbor not in visited:
                find_cycles(neighbor, path)

        path.pop()

    for node in graph:
        if node not in visited:
            find_cycles(node, [])

    return cycles[0] if cycles else []

def calculate_processed_time(wid, state):
    """计算晶圆已消耗的加工时间"""
    total = 0.0
    matid = next((m for w, m, _, _ in state.wafer_tasks if w == wid), None)
    if not matid:
        return None

    for move in state.move_list:
        if isinstance(move, dict) and move.get("MatID") == matid:
            total += move["EndTime"] - move["StartTime"]
    return total

def get_resource_priority(wid, state):
    current_module, _ = state.wafer_current_location.get(wid, (None, None))
    base_priority = GlobalConfig.RESOURCE_PRIORITY.get(current_module, 0.5)
    jit_urgency = calculate_jit_urgency(wid, get_now(state.start_time), state)

    # PM模块逻辑重构
    if current_module.startswith("PM"):
        status = state.pm_status[current_module]
        if status["current_wafer"] == wid:
            if status["in_use"]:  # 正在加工中
                return base_priority * 0.8  # 主动降低优先级
            else:  # 已完成加工待取出
                return base_priority * (1 + jit_urgency) * 1.2  # 增加取出权重
        elif not status["in_use"]:  # 等待进入PM
            return base_priority * (1 + jit_urgency)

    elif current_module.startswith("LL"):
        if current_module == "LLD":
            if "cooling" in state.ll_state.get(current_module, ""):
                # 情况1：正在冷却中 -> 降低优先级
                return base_priority * 0.7
            else:
                # 情况2：冷却完成待移出 -> 提高优先级
                return base_priority * 1.5
        # 其他LL（LLA/LLB/LLC）保持默认
    return base_priority * (1 + jit_urgency)


def calculate_jit_urgency(wid, now, state):
    """
    返回晶圆的JIT紧急程度（值越大越紧急，范围0~1）
    """
    if wid not in state.wafer_last_leave_time:
        return 0.0
    last_leave = state.wafer_last_leave_time[wid]
    elapsed = now - last_leave
    ratio = elapsed / (GlobalConfig.JIT_TRANSFER_LIMIT * GlobalConfig.TIME_SCALE)
    return min(1.0, max(0.0, ratio))

def get_original_lp(wid, state):
    """获取晶圆的原始LP模块"""
    for w, _, l_pname, _ in state.wafer_tasks:
        if w == wid:
            return l_pname
    return None


def find_last_ll(wid, state):
    """查找晶圆最近经过的LL模块"""
    # 先通过 wid 获取 matid
    matid = next((m for w, m, _, _ in state.wafer_tasks if w == wid), None)
    if not matid:
        return None

    # 倒序查找该晶圆最近一次经过的 LL 模块
    for move in reversed(state.move_list):
        if move["MatID"] == matid and move["ModuleName"].startswith("LL"):
            return move["ModuleName"]
    return None


def get_sacrifice_reason(wid, state):
    """获取牺牲晶圆的主要原因"""
    current_module, _ = state.wafer_current_location.get(wid, (None, None))

    reasons = []
    # 检查是否因为资源优先级低
    if current_module and GlobalConfig.RESOURCE_PRIORITY.get(current_module, 0.5) < 0.7:
        reasons.append(f"low-priority resource {current_module}")

    # 检查是否因为JIT紧急度低
    if wid in state.wafer_last_leave_time:
        elapsed = get_now(state.start_time) - state.wafer_last_leave_time[wid]
        if elapsed < GlobalConfig.JIT_TRANSFER_LIMIT * 0.5:
            reasons.append("low JIT urgency")

    return " & ".join(reasons) if reasons else "minimum total cost"


# ================= 辅助函数 =================
def get_now(start_time):
    """获取当前仿真时间"""
    return round(time.time() - start_time, 2)


def get_next_move_id(state):
    """生成唯一的移动ID"""
    with state.mutex_print:
        state.move_id_counter[0] += 1
        return state.move_id_counter[0] - 1

def tm2_move_time(m1, m2):
    """计算TM2移动时间"""
    if m1 == m2: return 0
    idx1, idx2 = GlobalConfig.tm2_ring.index(m1), GlobalConfig.tm2_ring.index(m2)
    dist = min((idx2 - idx1) % 8, (idx1 - idx2) % 8)
    return 4 * dist / 8


def tm3_move_time(m1, m2):
    """计算TM3移动时间"""
    if m1 == m2: return 0
    idx1, idx2 = GlobalConfig.tm3_ring.index(m1), GlobalConfig.tm3_ring.index(m2)
    dist = min((idx2 - idx1) % 8, (idx1 - idx2) % 8)
    return 4 * dist / 8


def choose_LL(state):
    """智能选择空闲LL口"""
    for ll in ["LLA", "LLB"]:
        with state.ll_slot_lock[ll]:
            s2_empty = state.ll_slots[ll]["S2"] is None
            s1_empty = state.ll_slots[ll]["S1"] is None
            if s2_empty and s1_empty:
                return ll
    return random.choice(["LLA", "LLB"])

def get_tm_candidate_wafers(state):
    candidates = []
    for wid, (module, _) in state.wafer_current_location.items():
        if module.startswith("PM"):
            status = state.pm_status[module]
            if status["current_wafer"] == wid and not status["in_use"]:
                # 已加工完成，等待取出
                priority = get_resource_priority(wid, state)
                candidates.append((wid, module, priority))
    # 按优先级从高到低排序
    candidates.sort(key=lambda x: -x[2])
    return candidates
def tm2_transfer_wafer(wid, matid, from_mod, to_mod,state):
    if from_mod == "LLD2": from_mod = "LLD"
    if to_mod == "LLD2": to_mod = "LLD"
    """TM2传输晶圆"""
    with state.sem_TM2:
        src_slot = 2 if from_mod in ["LLA", "LLB"] else 1
        # 检查手臂指向是否正确
        opposite = GlobalConfig.OPPOSITE_ARMS["TM2"].get(from_mod, None)
        if opposite:
            # 确保另一手臂没有冲突
            pass
        pick_time = GlobalConfig.PICK_TIME
        end_time = log_move("PickMove", "TM2", pick_time, state, matid, src_station=from_mod,
                            dest_station="TM2", src_slot=src_slot, dest_slot=1)

        # 计算移动时间
        move_time = tm2_move_time(from_mod, to_mod)
        end_time = log_move("TransferMove", "TM2", move_time, state, matid, src_station=from_mod,
                            dest_station=to_mod, src_slot=1, dest_slot=1)
        dest_slot = 2 if to_mod in ["LLA", "LLB"] else 1
        # 放晶圆时间
        place_time = GlobalConfig.PLACE_TIME
        end_time = log_move("PlaceMove", "TM2", place_time, state, matid, src_station="TM2",
                            dest_station=to_mod, src_slot=1, dest_slot=dest_slot)

        # 更新晶圆位置
        if from_mod.startswith("LL"):
            if from_mod in ["LLA", "LLB"]:
                state.ll_slots[from_mod]["S2"] = None
            else:
                state.ll_slots[from_mod]["S1"] = None
                state.ll_slots[from_mod]["S2"] = None
        current_time = get_now(state.start_time)
        check_jit_violation(wid, current_time, state, from_mod,"transfer")
        update_wafer_location(wid, to_mod, end_time, state)
        return end_time
def tm3_transfer_wafer(wid, matid, from_mod, to_mod, state):
    if from_mod == "LLD2": from_mod = "LLD"
    if to_mod == "LLD2": to_mod = "LLD"
    """TM3传输晶圆"""
    with state.sem_TM3:
        src_slot = 1
        # 检查手臂指向是否正确
        opposite = GlobalConfig.OPPOSITE_ARMS["TM3"].get(from_mod, None)
        if opposite:
            # 确保另一手臂没有冲突
            pass

        # 计算移动时间
        pick_time = GlobalConfig.PICK_TIME
        end_time = log_move("PickMove", "TM3", pick_time, state, matid, src_station=from_mod, dest_station="TM3",
            src_slot=src_slot, dest_slot=1)

        # 计算移动时间
        move_time = tm3_move_time(from_mod, to_mod)
        end_time = log_move("TransferMove", "TM3", move_time, state, matid, src_station=from_mod, dest_station=to_mod,
            src_slot=1, dest_slot=1)

        # 放晶圆时间
        dest_slot = 1
        place_time = GlobalConfig.PLACE_TIME
        end_time = log_move("PlaceMove", "TM3", place_time, state, matid, src_station="TM3", dest_station=to_mod,
            src_slot=1, dest_slot=dest_slot)

        # 更新晶圆位置
        if from_mod.startswith("LL"):
            if from_mod in ["LLA", "LLB"]:
                state.ll_slots[from_mod]["S2"] = None
            else:
                state.ll_slots[from_mod]["S1"] = None
                state.ll_slots[from_mod]["S2"] = None
        current_time = get_now(state.start_time)
        check_jit_violation(wid, current_time, state, from_mod,"transfer")
        update_wafer_location(wid, to_mod, end_time, state)
        return end_time

def pm_process_with_jit(pm_name, wid, process_time, matid, state,src_station=None, dest_station=None,
             src_slot=None, dest_slot=None):
    """PM加工流程（修复上下文管理器错误）"""
    # 获取PM状态（不使用with语句）
    status = state.pm_status[pm_name]
    # 检查PM是否可用
    while True:
        if not status["cleaning"] and not status["in_use"]:
            # 原子性更新状态
            status.update({
                "in_use": True,
                "current_wafer": wid,
                "start_time": time.time()
            })
            break
        time.sleep(0.1 * GlobalConfig.TIME_SCALE)

    # 记录加工开始
    start_time = get_now(state.start_time)
    time.sleep(process_time * GlobalConfig.TIME_SCALE)
    end_time = get_now(state.start_time)

    # 记录移动（不使用log_move以避免嵌套锁）
    with state.move_list_lock:
        state.move_list.append({
            "StartTime": start_time,
            "EndTime": end_time,
            "MoveID": get_next_move_id(state),
            "MoveType": state.move_type_dict["ProcessMove"],
            "ModuleName": pm_name,
            "MatID": matid,
            "SlotID": 1,
            "SrcSlotID": src_slot if src_slot is not None else 1,
            "DestSlotID": dest_slot if dest_slot is not None else 1,
            "SrcStation": src_station if src_station else "",
            "DestStation": dest_station if dest_station else ""
        })

    # 更新状态（需要保证原子性）
    status.update({
        "in_use": False,
        "current_wafer": None,
        "last_end_time": end_time,
        "count": status["count"] + 1,
        "idle_start": end_time
    })

    state.wafer_last_leave_time[wid] = end_time
    check_jit_violation(wid, end_time, state, pm_name,"process")
    update_wafer_location(wid, pm_name, end_time, state)
    return end_time
def ll_put_into_S2(ll_name, wid, state):
    """将晶圆放入LL的S2槽位"""
    with state.ll_slot_lock[ll_name]:
        state.ll_slots[ll_name]["S2"] = wid
        return True

def change_ll_state(ll_name, target_state, move_list, gantt_data, start_time, matid, state):
    """改变LL模块状态（大气/真空）"""
    current_state = state.ll_state[ll_name]
    if current_state == target_state:
        return True
    with state.ll_state_lock[ll_name]:
        if state.ll_state[ll_name] != target_state:
            if target_state == "vac":
                # 转换为真空状态
                duration = GlobalConfig.PUMP_TIME
                move_type = "PumpMove"
                src_slot = 1
                dest_slot = 2
            else:
                # 转换为大气状态
                duration = GlobalConfig.VENT_TIME
                move_type = "VentMove"
                src_slot = 2
                dest_slot = 1
            end_time = log_move(move_type, ll_name, duration, state, matid, src_station=ll_name, dest_station=ll_name, src_slot=src_slot, dest_slot=dest_slot)
            state.ll_state[ll_name] = target_state
            return end_time
    return True

def check_and_clean_pm(pm_name, current_path, move_list, gantt_data, start_time, state):
    status = state.pm_status[pm_name]
    now = get_now(start_time)
    clean_required = False
    clean_reason = ""
    clean_time = 0

    # 条件b: 优先检查工艺路径切换（优先级最高）
    if status["last_path"] and status["last_path"] != current_path and current_path != "system":
        clean_required = True
        clean_reason = "path_change"
        clean_time = GlobalConfig.CLEAN_CONDITIONS["path_change"]["time"]
        # 执行清洁后重置晶圆计数（新增）
        status["count"] = 0

        # 条件c: 加工晶圆数达到阈值（仅在未触发路径切换时检查）
    elif status["count"] >= GlobalConfig.CLEAN_CONDITIONS["wafer_count"]["threshold"]:
        clean_required = True
        clean_reason = "wafer_count"
        clean_time = GlobalConfig.CLEAN_CONDITIONS["wafer_count"]["time"]

    # 条件a: 空闲时间达到阈值（最低优先级）
    elif not clean_required:
        idle_time = now - status["idle_start"]
        if idle_time >= GlobalConfig.CLEAN_CONDITIONS["idle"]["threshold"]*GlobalConfig.TIME_SCALE:
            clean_required = True
            clean_reason = "idle"
            clean_time = GlobalConfig.CLEAN_CONDITIONS["idle"]["time"]

    # 执行清洁（如果满足任一条件且PM未被占用）
    if clean_required and not status["cleaning"]:
        status["cleaning"] = True
        log_move("CleanMove", pm_name, clean_time, state, "system", src_station=pm_name, dest_station=pm_name,src_slot=1, dest_slot=1)

        if clean_reason == "wafer_count":
            status["count"] = 0
        status["last_path"] = current_path
        clean_end_time = now + clean_time
        status["last_end_time"] = clean_end_time
        status["idle_start"] = clean_end_time
        status["cleaning"] = False
        return True
    return False

def ll_move_S2_to_S1(ll_name, state):
    """将晶圆从S2移动到S1"""
    with state.ll_slot_lock[ll_name]:
        state.ll_slots[ll_name]["S1"] = state.ll_slots[ll_name]["S2"]
        state.ll_slots[ll_name]["S2"] = None
        return True


def log_move(action, module, duration, state, matid,
             slot=1, src_station=None, dest_station=None,
             src_slot=None, dest_slot=None):
    t_start = None
    """记录移动操作"""
    try:
        t_start = get_now(state.start_time)
        time.sleep(duration * GlobalConfig.TIME_SCALE)
        t_end = get_now(state.start_time)

        with state.move_list_lock:
            state.move_list.append({
                "StartTime": t_start,
                "EndTime": t_end,
                "MoveID": get_next_move_id(state),
                "MoveType": state.move_type_dict.get(action, 3),
                "ModuleName": module,
                "MatID": matid,
                "SlotID": slot,
                "SrcSlotID": src_slot if src_slot is not None else 1,
                "DestSlotID": dest_slot if dest_slot is not None else 1,
                "SrcStation": src_station if src_station else "",
                "DestStation": dest_station if dest_station else ""
            })

        with state.gantt_data_lock:
            state.gantt_data.append((f"Wafer-{matid.split('.')[0]}", module, t_start, t_end))
        return t_end
    except Exception as e:
        with state.mutex_print:
            state.error_log.append((matid if matid else "system", f"Logging error at {module}: {str(e)}"))
        return t_start + duration if t_start is not None else duration


def update_wafer_location(wid, module, timestamp, state):
    """更新晶圆位置和最后离开时间"""
    state.wafer_current_location[wid] = (module, timestamp)
    state.wafer_last_leave_time[wid] = timestamp

def check_jit_violation(wid, current_time, state,pm_name,reason):
    """
    检查JIT违规并主动干预
    返回值: True表示已超时(需处理), False表示正常
    """
    if wid not in state.wafer_last_leave_time:
        return False

    current_module, _ = state.wafer_current_location.get(wid, (None, None))
    elapsed = current_time - state.wafer_last_leave_time[wid]
    if reason == "process":
        # 1. PM驻留时间限制（完成工艺后等待开门时间 ≤15s）
        if current_module and current_module.startswith("PM"):
            # 计算实际驻留时间（不包括必要的开门和取片时间）
            pm_status = state.pm_status[current_module]
            if pm_status["in_use"] and pm_status["current_wafer"] == wid:
                # 如果晶圆仍在PM中处理，不算驻留时间
                return False
            else:
                # 完成工艺后的等待时间（不包括开门和取片时间）
                state.CHECK_JIT_COUNT += 1
                if elapsed > GlobalConfig.JIT_PM_STAY_LIMIT*GlobalConfig.TIME_SCALE:
                    state.jit_violations.add(f"{wid}_{pm_name}")
                    return True
    elif reason == "transfer":
        # 2. 转移时间限制（完成当前节点后移动到下一节点 ≤30s）
        if pm_name.startswith("PM") and elapsed > GlobalConfig.JIT_TRANSFER_LIMIT*GlobalConfig.TIME_SCALE:
            state.CHECK_JIT_COUNT += 1
            state.jit_violations.add(f"{wid}_{pm_name}")
            return True

    return False

def evaluate_weighted_score(makespan, throughput, jit_violations, weights=(0.6, 0.3, 0.1)):
    """综合目标函数得分：越小越优"""
    w1, w2, w3 = weights
    return w1 * makespan + w2 * (-throughput) + w3 * jit_violations
# ================= 动态蚁群算法 =================
class DynamicACO:
    def __init__(self, state, num_ants=1, max_iterations=10):
        self.num_ants = num_ants  # 蚂蚁数量
        self.max_iterations = max_iterations  # 迭代次数
        self.state = state
        self.alpha = 1.0  # 信息素重要度初始值
        self.beta = 2.0  # 启发式因子重要度初始值
        self.evaporation = 0.1  # 信息素挥发率初始值
        self.last_improvement = time.time()
        self.ant_paths = {}
        # 历史记录
        self.history = {
            'time': [0.0],
            'alpha': [1.0],
            'beta': [2.0],
            'evaporation': [0.1],
            'load': [0.0],
            'jit_violations': [0.0],
            'deadlocks': [0.0]
        }
        self.data_lock = threading.Lock()  # 新增数据锁
        # 设置实时绘图
        plt.ion()  # 开启交互模式
        self.fig, self.axs = plt.subplots(3, 1, figsize=(12, 8))
        self.fig.suptitle('ACO Parameters Adaptation Process')

    def adapt_parameters(self):
        """增强型动态参数调整（线程安全 + last_improvement 自动更新）"""
        with self.data_lock:
            try:
                # 0. 仿真时间初始化检查
                if not hasattr(self.state, 'start_time'):
                    return

                # 1. 当前仿真时间
                current_time = get_now(self.state.start_time)

                # 2. 获取系统状态
                load = self.calculate_system_load()
                elapsed_time = max(0.1, current_time)  # 避免除零

                # 3. 系统指标标准化
                jit_violation_rate = min(1.0, len(self.state.jit_violations) / (self.state.CHECK_JIT_COUNT + 1e-6))
                deadlock_frequency = self.state.deadlock_count / max(1, current_time)

                # ===== JIT违规控制权重 =====
                jit_adjustment = 1.0
                if jit_violation_rate > 0.3:
                    jit_adjustment = 1.2

                # 4. 参数核心公式
                base_alpha = 1.0 + 0.6 * load + 0.4 * jit_violation_rate - 0.2 * deadlock_frequency
                base_beta = 2.0 - 0.5 * load
                base_evaporation = 0.1 + 0.1 * load + 0.05 * deadlock_frequency
                base_evaporation *= 0.8
                base_alpha *= jit_adjustment

                # 5. 动态探索机制（长期无提升则增加探索）
                exploration_boost = 0.0
                if not hasattr(self, 'last_improvement'):
                    self.last_improvement = current_time  # ✅ 初始化

                time_since_improvement = current_time - self.last_improvement
                if time_since_improvement > 10 * GlobalConfig.TIME_SCALE:
                    exploration_boost = min(0.5, 0.1 * (time_since_improvement / 10))
                    base_beta += exploration_boost
                    base_evaporation *= (1 + exploration_boost / 2)

                # 6. 应用参数并限制范围
                self.alpha = np.clip(base_alpha, 0.8, 2.0)
                self.beta = np.clip(base_beta, 1.0, 3.0)
                self.evaporation = np.clip(base_evaporation, 0.05, 0.3)

                # 7. 记录历史（统一仿真时间）
                self.history['time'].append(current_time)
                self.history['alpha'].append(self.alpha)
                self.history['beta'].append(self.beta)
                self.history['evaporation'].append(self.evaporation)
                self.history['load'].append(load)
                self.history['jit_violations'].append(jit_violation_rate)
                self.history['deadlocks'].append(deadlock_frequency)

                # 8. 若JIT违规率下降，更新 last_improvement
                if len(self.history['jit_violations']) >= 2:
                    prev = self.history['jit_violations'][-2]
                    curr = self.history['jit_violations'][-1]
                    if curr < prev:
                        self.last_improvement = current_time

                # 9. 控制绘图更新频率
                if len(self.history['time']) % max(1, GlobalConfig.PLOT_UPDATE_INTERVAL) == 0:
                    self._update_plots()

            except Exception as e:
                print(f"参数自适应异常: {str(e)}")
                self.alpha, self.beta, self.evaporation = 1.0, 2.0, 0.1

    def get_plot_data(self):
        """安全获取绘图数据"""
        with self.data_lock:
            # 确保所有数组长度一致且非空
            min_len = min(len(self.history[k]) for k in self.history)
            if min_len == 0:
                return None
            return {k: self.history[k][:min_len] for k in self.history}
    def save_adaptation_report(self, filename):
        """独立保存ACO参数报告"""
        plt.ioff()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.history['time'], self.history['alpha'], 'b-', label='Alpha')
        ax.plot(self.history['time'], self.history['beta'], 'r-', label='Beta')
        ax.plot(self.history['time'], self.history['evaporation'], 'g-', label='挥发率')
        ax.set_title('ACO参数自适应过程')
        ax.legend()
        ax.grid(True)
        fig.savefig(filename, dpi=300)
        plt.close(fig)
    def _update_plots(self):
        # 检查数据有效性
        if len(self.history['time']) < 2 or max(self.history['load']) <= 0:
            return
        # 设置合理的Y轴范围
        plt.ylim(0, 1.0)  # 负载率/JIT率范围[0,1]
        plt.yticks(np.arange(0, 1.1, 0.2))  # 固定刻度
        """更新实时图表"""
        # 清空旧图
        for ax in self.axs:
            ax.cla()

        # 子图1：核心参数变化
        self.axs[0].plot(self.history['time'], self.history['alpha'], label='Alpha')
        self.axs[0].plot(self.history['time'], self.history['beta'], label='Beta')
        self.axs[0].plot(self.history['time'], self.history['evaporation'], label='Evaporation')
        self.axs[0].set_title('ACO Core Parameters')
        self.axs[0].legend()
        self.axs[0].grid(True)

        # 子图2：系统负载指标
        self.axs[1].plot(self.history['time'], self.history['load'], 'b-', label='System Load')
        self.axs[1].plot(self.history['time'], self.history['jit_violations'], 'r--', label='JIT Violation Rate')
        self.axs[1].set_title('System Performance Indicators')
        self.axs[1].legend()
        self.axs[1].grid(True)
        # 子图3：异常事件
        self.axs[2].stem(self.history['time'], self.history['deadlocks'], 'g-', label='Deadlock Frequency')
        self.axs[2].set_title('Abnormal Events')
        self.axs[2].legend()
        self.axs[2].grid(True)

        # 调整布局
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)

    def calculate_system_load(self):
        """
        计算系统负载：
        - 综合PM、LL、TM资源的忙碌比例
        - 不使用队列，使用 in_use / occupied 状态
        """
        busy_pm = sum(1 for pm in self.state.pm_status.values() if pm["in_use"])
        total_pm = len(self.state.pm_status)

        busy_tm = sum(1 for tm in ["TM1", "TM2", "TM3"] if self.state.sem_TM1._value == 0)
        total_tm = 3

        busy_ll = 0
        total_ll = 0
        for ll_name, slots in self.state.ll_slots.items():
            for slot in ["S1", "S2"]:
                total_ll += 1
                if slots[slot] is not None:
                    busy_ll += 1

        # 计算资源占用率
        pm_load = busy_pm / total_pm if total_pm > 0 else 0
        tm_load = busy_tm / total_tm if total_tm > 0 else 0
        ll_load = busy_ll / total_ll if total_ll > 0 else 0

        # 平均负载（加权）
        load = 0.5 * pm_load + 0.3 * ll_load + 0.2 * tm_load
        return round(load, 3)

    def select_pm(self, candidates, urgency, wid=None):
        try:
            # ===== 1. 参数预处理 =====
            if not candidates:
                raise ValueError("Empty candidate PM list")

            # 获取当前仿真时间
            now = get_now(self.state.start_time)

            # 获取当前蚂蚁ID（线程安全）
            ant_id = threading.get_ident()
            with self.data_lock:
                if ant_id not in self.ant_paths:
                    self.ant_paths[ant_id] = set()

            # ===== 2. 动态参数调整 =====
            self.adapt_parameters()

            # ===== 3. 多蚂蚁探索 =====
            ant_scores = []
            for ant_idx in range(self.num_ants):
                # 3.1 过滤已访问节点（线程安全访问）
                with self.data_lock:
                    visited = self.ant_paths[ant_id].copy()

                valid_pms = [pm for pm in candidates if pm not in visited]

                # 如果所有候选都被访问过，重置路径（探索-利用平衡）
                if not valid_pms:
                    valid_pms = candidates
                    with self.data_lock:
                        self.ant_paths[ant_id] = set()

                # 3.2 计算各PM的吸引力得分
                scores = []
                total_score = 0.0

                for pm in valid_pms:
                    try:
                        # 获取PM状态（线程安全）
                        with self.state.mutex_print:
                            status = self.state.pm_status.get(pm, {})
                            if not status:
                                continue

                            pheromone = self.get_pheromone(pm)
                            pm_in_use = status.get("in_use", False)
                            pm_cleaning = status.get("cleaning", False)
                            last_end_time = status.get("last_end_time", now)

                        # 计算时间延迟（添加10%随机扰动模拟蚂蚁感知差异）
                        time_delay = max(0, (max(last_end_time, now) - now))
                        perturbed_delay = time_delay * random.uniform(0.9, 1.1)

                        # ---- 启发式函数计算 ----
                        # 空闲奖励（未使用且未清洁）
                        idle_bonus = 1.0 if (not pm_in_use and not pm_cleaning) else 0.1

                        # 时间可用性评分（延迟越小分数越高）
                        time_score = 1.0 / (1.0 + perturbed_delay)

                        # JIT紧急度调整（当前晶圆特定或通用紧急度）
                        if wid:
                            base_jit = calculate_jit_urgency(wid, now, self.state)
                        else:
                            base_jit = urgency
                        jit_level = base_jit * random.uniform(0.95, 1.05)  # 5%感知差异

                        # 综合启发值
                        heuristic = idle_bonus * time_score * (1 + jit_level)

                        # ---- 综合吸引力 ----
                        score = (pheromone ** self.alpha) * (heuristic ** self.beta)
                        scores.append((pm, score))
                        total_score += score

                    except Exception as e:
                        print(f"PM {pm} scoring error: {str(e)}")
                        continue

                # 3.3 蚂蚁决策（80%概率选择/20%探索）
                if scores and total_score > 0:
                    if random.random() < 0.8:  # 利用已知信息
                        # 轮盘赌选择
                        probs = [s[1] / total_score for s in scores]
                        chosen_idx = np.random.choice(len(scores), p=probs)
                        chosen_pm = scores[chosen_idx][0]
                    else:  # 随机探索
                        chosen_pm = random.choice(valid_pms)

                    # 记录选择（线程安全）
                    with self.data_lock:
                        self.ant_paths[ant_id].add(chosen_pm)

                    ant_scores.append((chosen_pm, total_score))

            # ===== 4. 蚁群决策 =====
            if ant_scores:
                # 选择总评分最高的PM
                best_pm = max(ant_scores, key=lambda x: x[1])[0]

                # 更新信息素（线程安全）
                self.update_pheromone(best_pm)
                return best_pm

            # 默认回退：随机选择（理论上不应执行到这里）
            return random.choice(candidates)

        except Exception as e:
            print(f"ACO selection error: {str(e)}")
            # 异常恢复：随机选择+重置蚂蚁路径
            with self.data_lock:
                self.ant_paths[ant_id] = set()
            return random.choice(candidates)

    def compute_reward(self):
        """根据当前状态评估一个多目标综合评分（用于更新信息素）"""
        if not self.state.wafer_last_leave_time:
            return 1.0  # 避免除零

        makespan = max(self.state.wafer_last_leave_time.values())
        throughput = len(self.state.wafer_last_leave_time) / makespan if makespan > 0 else 0
        jit_violations = len(self.state.jit_violations)

        # 可使用动态权重策略
        if jit_violations > 100:
            weights = (0.5, 0.2, 0.3)
        else:
            weights = (0.7, 0.2, 0.1)

        score = evaluate_weighted_score(makespan, throughput, jit_violations, weights)
        return max(score, 1.0)  # 避免分母为0
    def get_pheromone(self, pm):
        """获取PM的信息素值"""
        for group in GlobalConfig.pheromone_table:
            if pm in group:
                return GlobalConfig.pheromone_table[group][pm]
        return 1.0

    def update_pheromone(self, pm, duration=None):
        """更新信息素（使用 reward 代替 duration）"""
        reward = self.compute_reward()  # 综合多目标打分
        for group in GlobalConfig.pheromone_table:
            if pm in group:
                # 信息素挥发
                for m in group:
                    GlobalConfig.pheromone_table[group][m] *= (1 - self.evaporation)
                # 奖励当前选择
                boost = GlobalConfig.pheromone_boost / reward
                GlobalConfig.pheromone_table[group][pm] += boost
                self.last_improvement = time.time()
                break


def _update_real_time_plots(axs, state, aco):
    """线程安全的实时绘图更新"""
    try:
        plot_data = aco.get_plot_data()
        if plot_data is None or len(plot_data['time']) < 2:  # 至少需要2个点才能绘图
            return
        # 清除旧图（保持坐标轴不变）
        for ax in axs:
            ax.cla()
        # 子图1：ACO核心参数
        axs[0].clear()
        axs[0].plot(aco.history['time'], aco.history['alpha'], 'b-', label='信息素权重')
        axs[0].plot(aco.history['time'], aco.history['beta'], 'r-', label='启发式权重')
        axs[0].set_ylabel('参数值')
        axs[0].legend(loc='upper left')

        # 子图2：系统指标（共享x轴）
        axs[1].clear()
        axs[1].plot(aco.history['time'], aco.history['load'], 'g-', label='系统负载')
        axs[1].plot(aco.history['time'], aco.history['jit_violations'], 'm--', label='JIT违规率')
        axs[1].set_xlabel('时间(s)')
        axs[1].set_ylabel('指标值')
        axs[1].legend(loc='upper left')

        ax2 = axs[1].twinx()
        ax2.stem(aco.history['time'], aco.history['deadlocks'], 'r:', label='死锁事件')
        ax2.set_ylabel('死锁次数')
        ax2.legend(loc='upper right')

        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)

    except Exception as e:
        print(f"绘图更新失败: {str(e)}")
        # 失败后自动关闭绘图避免阻塞
        plt.close('all')
class HierarchicalScheduler:
    def __init__(self, state):
        self.state = state
        self.aco = DynamicACO(state)
        self.last_adjust_time = 0
        self.adjust_interval = 5.0
        self.plot_update_counter = 0  # 新增绘图计数器
    def global_schedule(self, wafer, path_type):
        """全局调度决策"""
        current_module = wafer["current_stage"]
        next_module = None

        # 确定下一步模块
        if current_module.startswith("LP"):
            next_module = "AL"
        elif current_module == "AL":
            next_module = choose_LL(self.state)
        else:
            # 根据路径模板确定候选模块
            path = GlobalConfig.path_templates[path_type]
            for i, mod in enumerate(path):
                if mod == current_module or isinstance(mod, list) and current_module in mod:
                    next_module = path[i+1] if i+1 < len(path) else None
                    break

        # 如果是PM组，使用蚁群算法选择具体PM
        if isinstance(next_module, list):
            # 计算JIT紧急程度
            current_time = get_now(self.state.start_time)
            elapsed = current_time - wafer["last_move_time"]
            urgency = min(1.0, elapsed / (GlobalConfig.JIT_TRANSFER_LIMIT * 0.8))
            wid = wafer["id"]
            # 使用ACO选择PM
            chosen_pm = self.aco.select_pm(next_module, urgency, wid)
            return chosen_pm
        return next_module

    def local_adjust(self):
        """局部调整"""
        current_time = get_now(self.state.start_time)
        if current_time - self.last_adjust_time < self.adjust_interval:
            return

        self.last_adjust_time = current_time
        # 1. 执行ACO参数自适应（自动记录历史数据）
        for iteration in range(self.aco.max_iterations):
            self.aco.adapt_parameters()  # 动态调整参数

        # 2. 每10次调整触发绘图更新（降低性能开销）
        self.plot_update_counter += 1
        if self.plot_update_counter % 10 == 0:
            self.aco._update_plots()
        # 3. 主动触发PM,idle清洁检查（每5秒触发一次）
        if current_time - self.state.last_clean_check_time >= self.state.clean_check_interval*GlobalConfig.TIME_SCALE:
            self.state.last_clean_check_time = current_time
            for pm_name in self.state.pm_status:
                check_and_clean_pm(
                    pm_name,
                    current_path="system",
                    move_list=self.state.move_list,
                    gantt_data=self.state.gantt_data,
                    start_time=self.state.start_time,
                    state=self.state
                )
        # 检查并解决资源冲突
        self.resolve_conflicts()
        # 死锁预防 - 打破潜在的死锁条件
        self.prevent_deadlocks()
        # 平衡PM负载
        self.balance_pm_loads()
    def resolve_conflicts(self):
        """解决资源冲突（修改版：无队列逻辑）"""
        current_time = get_now(self.state.start_time)

        for wid, (current_module, last_move_time) in list(self.state.wafer_current_location.items()):
            elapsed = current_time - last_move_time

            # 检查JIT预警（80%阈值）
            if elapsed > GlobalConfig.JIT_TRANSFER_LIMIT * 0.8:
                # 情况1：晶圆正在PM中加工
                if current_module.startswith("PM"):
                    with self.state.pm_status[current_module] as status:
                        if status["current_wafer"] == wid:
                            # 标记为紧急晶圆（让PM尽快完成加工）
                            status["urgent"] = True
                            continue

                # 情况2：晶圆在LL等待传输
                elif current_module.startswith("LL"):
                    # 提升LL槽位优先级（供机械手优先选择）
                    with self.state.ll_slot_lock[current_module]:
                        if wid in [self.state.ll_slots[current_module]["S1"],
                                   self.state.ll_slots[current_module]["S2"]]:
                            self.state.ll_slots[current_module]["priority"] = max(
                                self.state.ll_slots[current_module].get("priority", 0),
                                int(elapsed / GlobalConfig.JIT_TRANSFER_LIMIT * 10)  # 动态优先级
                            )
    def balance_pm_loads(self):
        """平衡PM负载（修改版：无队列逻辑）"""
        # 1. 计算PM使用率（基于加工时间占比）
        pm_stats = {}
        current_time = get_now(self.state.start_time)

        for pm, status in self.state.pm_status.items():
            # 计算最近利用率 = 加工时间 / (当前时间 - 启动时间)
            if status["start_time"] > 0:
                usage_ratio = (status["last_end_time"] - status["start_time"]) / \
                              (current_time - status["start_time"])
            else:
                usage_ratio = 0.0

            pm_stats[pm] = {
                "in_use": status["in_use"],
                "usage": min(1.0, max(0.0, usage_ratio)),  # 限制在0~1范围
                "last_path": status["last_path"]
            }

        # 2. 计算平均利用率
        if pm_stats:
            avg_usage = sum(s["usage"] for s in pm_stats.values()) / len(pm_stats)
        else:
            avg_usage = 0.5  # 默认值

        # 3. 动态调整ACO参数
        overload_pms = [pm for pm, s in pm_stats.items()
                        if s["usage"] > avg_usage + 0.2]  # 使用率超过平均值20%

        if overload_pms:
            # 提高过载PM的信息素挥发率
            self.aco.evaporation = min(0.3, self.aco.evaporation + 0.05)
    def prevent_deadlocks(self):
        """死锁预防措施"""
        # 检查是否有晶圆等待时间过长
        current_time = get_now(self.state.start_time)
        for wid, (mod, t) in list(self.state.wafer_current_location.items()):
            elapsed = current_time - t
            if elapsed > GlobalConfig.JIT_TRANSFER_LIMIT * 0.5:  # 比JIT限制更严格的条件
                # 尝试重新调度
                self.force_reschedule(wid)
    def force_reschedule(self, wid):
        """强制重新调度晶圆（修改版：无队列逻辑）"""
        current_module, last_time = self.state.wafer_current_location.get(wid, (None, None))
        if not current_module:
            return

        current_time = get_now(self.state.start_time)
        elapsed = current_time - last_time if last_time else float('inf')

        # 情况1：晶圆正在PM中加工
        if current_module.startswith("PM"):
            with self.state.pm_status[current_module] as status:
                if status["current_wafer"] == wid:
                    # 标记为紧急任务（让PM尽快完成加工）
                    status["urgent"] = True
                    # 缩短剩余加工时间（如有必要）
                    if elapsed > GlobalConfig.JIT_PM_STAY_LIMIT * 0.8:
                        status["estimated_remain"] = max(1, status.get("estimated_remain", 0) * 0.8)
                    return

        # 情况2：晶圆在LL等待传输
        elif current_module.startswith("LL"):
            # 提升该LL槽位的优先级
            with self.state.ll_slot_lock[current_module]:
                for slot in ["S1", "S2"]:
                    if self.state.ll_slots[current_module][slot] == wid:
                        # 动态计算优先级（等待越久优先级越高）
                        priority = min(10, int(elapsed / GlobalConfig.JIT_TRANSFER_LIMIT * 15))
                        self.state.ll_slots[current_module]["priority"] = priority
                        break

def plot_optimization_trend(history_total_time, history_throughput, history_jit_violation):
    """
    绘制多目标优化趋势图：f1（总时间）、f2（吞吐量）、f3（JIT违规）
    """
    if not history_total_time or not history_throughput or not history_jit_violation:
        print("无足够数据绘制优化趋势图")
        return

    iteration_list = list(range(len(history_total_time)))

    plt.figure(figsize=(12, 6))

    # 绘制目标函数变化曲线
    plt.plot(iteration_list, history_total_time, label='Total Time (f1)', marker='o')
    plt.plot(iteration_list, history_throughput, label='Throughput (f2)', marker='s')
    plt.plot(iteration_list, history_jit_violation, label='JIT Violations (f3)', marker='^')

    plt.xlabel("Iteration")
    plt.ylabel("Objective Value")
    plt.title("Multi-objective Optimization Trend (Ant Colony)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("optimization_trend_1.png")
    plt.show()

def draw_gantt(gantt_data, task):
    """绘制甘特图"""
    if not gantt_data:
        print("无甘特图数据可绘制")
        return

    fig, ax = plt.subplots(figsize=(14, 10))
    module_labels = list(sorted(set([mod for _, mod, _, _ in gantt_data]),
                                key=lambda x: ("LP" in x, "AL" in x, "LL" in x, "PM" in x, "TM" in x)))
    module_idx = {mod: i for i, mod in enumerate(module_labels)}

    colors = plt.cm.tab20(np.linspace(0, 1, len(set([w for w, _, _, _ in gantt_data]))))
    wafer_colors = {w: colors[i] for i, w in enumerate(set([w for w, _, _, _ in gantt_data]))}

    for wafer, module, start, end in gantt_data:
        y = module_idx[module]
        ax.barh(y, end - start, left=start, height=0.4, color=wafer_colors[wafer])

    ax.set_yticks(list(module_idx.values()))
    ax.set_yticklabels(module_labels)
    ax.set_xlabel("时间（秒）")
    ax.set_title("晶圆调度甘特图")
    plt.tight_layout()
    if task == "a":
        plt.savefig("gantt_chart_1a.png")
    elif task == "b":
        plt.savefig("gantt_chart_1b.png")
    elif task == "c":
        plt.savefig("gantt_chart_1c.png")
    else:
        plt.savefig("gantt_chart_1d.png")
    plt.show()


def output_error_log(state):
    """输出包含牺牲统计的错误日志"""
    if state.error_log:
        print("\n=== 错误日志 ===")
        for wid, errmsg in state.error_log:
            print(f"晶圆 {wid}: {errmsg}")

    print("\n=== 死锁解决统计 ===")
    print(f"总解决次数: {state.sacrifice_stats['total']}")
    print("按原因分类:")
    for reason, count in state.sacrifice_stats['by_reason'].items():
        print(f"- {reason}: {count}次")


def wait_for_prev_lp_started(lp_name, state):
    if lp_name != "LP1":
        lp_index = int(lp_name[2:])
        prev_lp = f"LP{lp_index - 1}"
        prev_lp_wids = [w[0] for w in state.wafer_tasks if w[2] == prev_lp]
        while not all(
                wid2 in state.wafer_current_location and
                state.wafer_current_location[wid2][0] != prev_lp
                for wid2 in prev_lp_wids
        ):
            pending = [wid2 for wid2 in prev_lp_wids
                       if wid2 not in state.wafer_current_location or
                       state.wafer_current_location[wid2][0] == prev_lp]
            # print(f"等待晶圆盒 {prev_lp} 启动加工: 尚未离开晶圆 {pending}")
            time.sleep(1)
def wafer_process(wid, matid, lp_name, path_type, state, scheduler):
    """晶圆处理主流程（修改路径选择逻辑）"""
    wafer = {
        "id": wid,
        "matid": matid,
        "current_stage": lp_name,
        "last_move_time": get_now(state.start_time),
        "path_type": path_type
    }
    # 记录初始位置
    update_wafer_location(wid, lp_name, wafer["last_move_time"], state)
    # a和b任务模式：按晶圆盒编号依次加工，当未完成加工的晶圆都离开当前晶圆盒后，下一个晶圆盒内的晶圆才能被调度。
    if path_type in {"A", "B"}:
        wait_for_prev_lp_started(lp_name, state)
    # --- 出片阶段 ---
    # 从LP取出晶圆
    with state.lp_sequence_lock[lp_name]:
        wafer_num = int(matid.split('.')[1])
        while wafer_num != state.lp_next_expected[lp_name]:
            state.lp_sequence_cond[lp_name].wait()
        # ===== 真正执行取片动作 =====
        with state.sem_LP[lp_name]:
            log_move("PrepareMove", lp_name, GlobalConfig.OPEN_DOOR_TIME, state, matid,src_station=lp_name, dest_station=lp_name,src_slot=1,dest_slot=1)
            with state.sem_TM1:
                log_move("PickMove", "TM1", GlobalConfig.PICK_TIME, state, matid,src_station=lp_name,dest_station="TM1",src_slot=1,dest_slot=1)
                log_move("TransferMove", "TM1", GlobalConfig.TRANSFER_TIME_TM1, state, matid,src_station=lp_name,
                         dest_station="AL", src_slot=1, dest_slot=1)
            log_move("CompleteMove", lp_name, GlobalConfig.CLOSE_DOOR_TIME, state, matid, src_station=lp_name, dest_station=lp_name,src_slot=1,dest_slot=1)
        # ===== 更新下一个应出片编号 =====
        state.lp_next_expected[lp_name] += 1
        state.lp_sequence_cond[lp_name].notify_all()

    # AL对准
    with state.sem_AL:
        end_time = log_move("AlignMove", "AL", GlobalConfig.ALIGN_TIME, state, matid, src_station="AL", dest_station="AL", src_slot=1, dest_slot=1)
        update_wafer_location(wid, "AL", end_time, state)
        wafer["last_move_time"] = end_time
    # --- 进入加工区 ---
    # 选择LL入口
    ll_start = choose_LL(state)
    wafer["current_stage"] = ll_start

    # AL到LL (需要先转换为真空状态)
    change_ll_state(ll_start, "vac", state.move_list, state.gantt_data, state.start_time, matid, state)

    with state.sem_TM1:
        log_move("TransferMove", "TM1", GlobalConfig.TRANSFER_TIME_TM1, state, matid,src_station="AL",dest_station=ll_start,src_slot=1,dest_slot=1)

    # 放入LL的S2槽位
    ll_put_into_S2(ll_start, wid, state)
    log_move("PrepareMove", ll_start, GlobalConfig.OPEN_DOOR_TIME, state, matid, slot=1, src_station=ll_start,
             dest_station=ll_start, src_slot=1, dest_slot=1)
    with state.sem_LL[ll_start]:
        log_move("PlaceMove", ll_start, GlobalConfig.PLACE_TIME, state, matid, slot=2, src_station=ll_start,
                 dest_station=ll_start, src_slot=1, dest_slot=2)
    log_move("CompleteMove", ll_start, GlobalConfig.CLOSE_DOOR_TIME, state, matid, slot=2, src_station=ll_start,
             dest_station=ll_start, src_slot=2, dest_slot=2)

    # 从S2移动到S1
    ll_move_S2_to_S1(ll_start, state)
    update_wafer_location(wid, ll_start, get_now(state.start_time), state)
    wafer["last_move_time"] = get_now(state.start_time)

    # --- 加工阶段 ---
    # 根据任务2a选择不同的路径模板
    path_template = GlobalConfig.path_templates[path_type]
    path = [ll_start if p == "LLA" else p for p in path_template]
    for i in range(1, len(path) - 1):  # 跳过第一个和最后一个LL
        current_module = path[i]
        if isinstance(current_module, list) or (isinstance(current_module,str) and current_module.startswith("PM")):
            # 使用全局调度选择PM
            last = wafer['current_stage']
            if isinstance(current_module,str):
                chosen_pm = current_module
            else:
                chosen_pm = scheduler.global_schedule(wafer, path_type)
                if chosen_pm is None:
                    chosen_pm = current_module[0]
                if chosen_pm in ["LLC", "LLD"]:
                    print("=========选择错误========")
            if last in ["LLA", "LLB", "LLC", "LLD"]:
                if chosen_pm in ["PM1", "PM2", "PM3", "PM4", "PM5", "PM6"]:
                    with state.ll_slot_lock[last]:
                        state.ll_slots[last]["S1"] = None
                        state.ll_slots[last]["S2"] = None
                else:
                    if last in ["LLA", "LLB"]:
                        with state.ll_slot_lock[last]:
                            state.ll_slots[last]["S2"] = None
                    else:
                        with state.ll_slot_lock[last]:
                            state.ll_slots[last]["S1"] = None
                            state.ll_slots[last]["S2"] = None
            # 等待进入PM队列
            # wait_in_pm_queue(chosen_pm, wid, state.start_time, state)
            with state.sem_PM[chosen_pm]:
                # 检查并执行清洁
                check_and_clean_pm(chosen_pm, path_type, state.move_list, state.gantt_data, state.start_time, state)

                # 开门
                log_move("PrepareMove", chosen_pm, GlobalConfig.OPEN_DOOR_TIME, state, matid,src_station=chosen_pm,dest_station=chosen_pm,src_slot=1,dest_slot=1)

                # 使用TM传输到PM
                if chosen_pm in ["PM7", "PM8", "PM9", "PM10"]:
                    # 使用TM2传输
                    end_time = tm2_transfer_wafer(wid, matid, wafer["current_stage"], chosen_pm,state)
                elif chosen_pm in ["PM1", "PM2", "PM3", "PM4", "PM5", "PM6"]:
                    # 使用TM3传输
                    end_time = tm3_transfer_wafer(wid, matid, wafer["current_stage"], chosen_pm, state)
                update_wafer_location(wid, chosen_pm, end_time, state)
                wafer["current_stage"] = chosen_pm
                wafer["last_move_time"] = end_time
                # 关门
                log_move("CompleteMove", chosen_pm, GlobalConfig.CLOSE_DOOR_TIME, state, matid, src_station=chosen_pm,dest_station=chosen_pm,src_slot=1,dest_slot=1)
                # 加工
                ptime = GlobalConfig.PROCESS_TIMES.get(chosen_pm, 100)
                end_time = pm_process_with_jit(chosen_pm, wid, ptime, matid, state)
                # 更新PM状态

                state.pm_status[chosen_pm]["last_path"] = path_type
                state.pm_status[chosen_pm]["count"] += 1
                state.pm_status[chosen_pm]["last_end_time"] = end_time
                state.pm_status[chosen_pm]["idle_start"] = end_time

                # 更新信息素
                scheduler.aco.update_pheromone(chosen_pm, ptime)
                wafer["last_move_time"] = end_time
                while True:
                    self_priority = get_resource_priority(wid, state)
                    others = get_tm_candidate_wafers(state)
                    more_urgent = [w for w in others if w[0] != wid and w[2] > self_priority]
                    if not more_urgent:
                        break
                    time.sleep(0.2 * GlobalConfig.TIME_SCALE)
        elif isinstance(current_module, str) and current_module in ["LLA", "LLB"]:
            last = wafer["current_stage"]
            if last in ["LLC", "LLD"]:
                with state.ll_slot_lock[last]:
                    state.ll_slots[last]["S1"] = None
                    state.ll_slots[last]["S2"] = None
            # 需要状态转换
            change_ll_state(current_module, "atm", state.move_list, state.gantt_data,
                            state.start_time, matid, state)
            if last in ["PM7", "PM8", "PM9", "PM10", "LLA", "LLB", "LLC", "LLD"]:
                tm2_transfer_wafer(wid, matid, wafer["current_stage"], current_module, state)
            else:
                print("无法移动")
            # 放入LL
            ll_put_into_S2(current_module, wid, state)
            log_move("PrepareMove", current_module, GlobalConfig.OPEN_DOOR_TIME, state, matid, slot=2,
                     src_station=current_module, dest_station=current_module, src_slot=2, dest_slot=2)
            with state.sem_LL[current_module]:
                log_move("PlaceMove", current_module, GlobalConfig.PLACE_TIME, state, matid, slot=2,
                         src_station=last, dest_station=current_module, src_slot=1, dest_slot=2)
            log_move("CompleteMove", current_module, GlobalConfig.CLOSE_DOOR_TIME, state, matid, slot=2,
                     src_station=current_module, dest_station=current_module, src_slot=2, dest_slot=2)
            if current_module in ["LLA", "LLB"]:
                # 从S2移动到S1
                ll_move_S2_to_S1(current_module, state)
            update_wafer_location(wid, current_module, get_now(state.start_time), state)
            wafer["current_stage"] = current_module
            wafer["last_move_time"] = get_now(state.start_time)
        elif isinstance(current_module, str) and current_module in ["LLC", "LLD", "LLD2"]:
            last = wafer["current_stage"]
            if last in ["PM1", "PM2", "PM3", "PM4", "PM5", "PM6", "LLC", "LLD"]:
                tm3_transfer_wafer(wid, matid, last, current_module,state)
            elif last in ["PM7", "PM8", "PM9", "PM10", "LLA", "LLB", "LLC", "LLD"]:
                tm2_transfer_wafer(wid, matid, last, current_module,state)
            if current_module == "LLD2":
                with state.ll_slot_lock["LLD"]:
                    state.ll_slots["LLD"]["S1"] = wid
                    state.ll_slots["LLD"]["S2"] = wid
                log_move("PrepareMove", "LLD", GlobalConfig.OPEN_DOOR_TIME, state, matid, slot=1, src_station="LLD",
                         dest_station="LLD", src_slot=1, dest_slot=1)
                with state.sem_LL["LLD"]:
                    log_move("PlaceMove", "LLD", GlobalConfig.PLACE_TIME, state, matid, slot=1, src_station=last,
                             dest_station="LLD", src_slot=1, dest_slot=1)
                log_move("ProcessMove", "LLD", GlobalConfig.COOL_TIME, state, matid, src_station="LLD",
                         dest_station="LLD", src_slot=1, dest_slot=1)
                log_move("CompleteMove", "LLD", GlobalConfig.CLOSE_DOOR_TIME, state, matid, slot=1, src_station="LLD",
                         dest_station="LLD", src_slot=1, dest_slot=1)
                update_wafer_location(wid, "LLD", get_now(state.start_time), state)
                wafer["current_stage"] = "LLD"
                end_time = get_now(state.start_time)
                wafer["last_move_time"] = end_time
            else:
                with state.ll_slot_lock[current_module]:
                    state.ll_slots[current_module]["S1"] = wid
                    state.ll_slots[current_module]["S2"] = wid
                log_move("PrepareMove", current_module, GlobalConfig.OPEN_DOOR_TIME, state, matid, slot=1,
                         src_station=current_module, dest_station=current_module, src_slot=1, dest_slot=1)
                with state.sem_LL[current_module]:
                    log_move("PlaceMove", current_module, GlobalConfig.PLACE_TIME, state, matid, slot=1,
                             src_station=last, dest_station=current_module, src_slot=1, dest_slot=1)
                log_move("CompleteMove", current_module, GlobalConfig.CLOSE_DOOR_TIME, state, matid, slot=1,
                         src_station=current_module, dest_station=current_module, src_slot=1, dest_slot=1)
                update_wafer_location(wid, current_module, get_now(state.start_time), state)
                wafer["current_stage"] = current_module
                end_time = get_now(state.start_time)
                wafer["last_move_time"] = end_time
    if not (wafer["current_stage"] in ["LLA", "LLB"]):
        with state.sem_TM2:
            # 取出晶圆
            last = wafer["current_stage"]
            if last in ["LLC", "LLD"]:
                with state.ll_slot_lock[last]:
                    state.ll_slots[last]["S1"] = None
                    state.ll_slots[last]["S2"] = None
            log_move("PickMove", wafer["current_stage"], GlobalConfig.PICK_TIME, state, matid, slot=1,src_station=last,dest_station="TM2",src_slot=1,dest_slot=1)
            update_wafer_location(wid, "TM2", get_now(state.start_time), state)

            # 2. 传输到LLA/LLB（选择空闲的LL口）
            ll_exit = choose_LL(state)  # 自动选择LLA或LLB
            change_ll_state(ll_exit, "atm", state.move_list, state.gantt_data, state.start_time, matid, state)
            # TM2移动时间计算
            move_time = tm2_move_time(wafer["current_stage"], ll_exit)
            log_move("TransferMove", "TM2", move_time, state, matid, slot=2, src_station=wafer["current_stage"],
                     dest_station=ll_exit, src_slot=1, dest_slot=2)

            # 放入LLA/LLB的S2槽
            log_move("PrepareMove", ll_exit, GlobalConfig.OPEN_DOOR_TIME, state, matid, slot=2, src_station=ll_exit,
                     dest_station=ll_exit, src_slot=2, dest_slot=2)
            ll_put_into_S2(ll_exit, wid, state)
            ll_move_S2_to_S1(ll_exit, state)
            with state.sem_LL[ll_exit]:
                log_move("PlaceMove", ll_exit, GlobalConfig.PLACE_TIME, state, matid, slot=2, src_station="TM2",
                         dest_station=ll_exit, src_slot=1, dest_slot=2)
            log_move("CompleteMove", ll_exit, GlobalConfig.CLOSE_DOOR_TIME, state, matid, slot=2, src_station=ll_exit,
                     dest_station=ll_exit, src_slot=2, dest_slot=2)

            update_wafer_location(wid, ll_exit, get_now(state.start_time), state)
    # 3. 从LLA/LLB到LP（使用TM1大气机械手）
    with state.sem_TM1:
        log_move("PickMove", "TM1", GlobalConfig.PICK_TIME, state, matid, src_station=ll_exit, dest_station="TM1",
                 src_slot=1, dest_slot=1)
        log_move("TransferMove", "TM1", GlobalConfig.TRANSFER_TIME_TM1, state, matid, src_station=ll_exit,
                 dest_station=lp_name, src_slot=1, dest_slot=1)
        log_move("PlaceMove", "TM1", GlobalConfig.PLACE_TIME, state, matid, src_station="TM1", dest_station=lp_name,
                 src_slot=1, dest_slot=1)
        # 4. 放回LP
    with state.sem_LP[lp_name]:
        log_move("CompleteMove", lp_name, GlobalConfig.CLOSE_DOOR_TIME, state, matid, src_station=lp_name,
                 dest_station=lp_name, src_slot=1, dest_slot=1)
    # 更新最终位置
    update_wafer_location(wid, lp_name, get_now(state.start_time), state)
