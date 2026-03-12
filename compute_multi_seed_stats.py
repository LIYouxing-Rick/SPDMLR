import torch
import os
from collections import defaultdict
import numpy as np
import argparse
import re

def parse_filename(filename):
    """
    解析文件名，提取关键参数（增强版，支持 nLR/lLR 前缀与架构片段连接，兼容旧/新命名）。

    示例：
      1024-net0.005-loss0.0005-LAM-AMSGRAD-0--20-16-8-P25-olm-LPon-LOGMon-Pow1.00_12_34_56
      0.001-0.0001-nLR0.001-lLR0.01-TSMNet+SPDDSMBNLogEigMLR-[40,20]-P100-olm-L1.00-G0.00-nLR0.001-lLR0.01_seed1

    Returns
    -------
    dict
        包含各参数的字典
    """
    try:
        raw_filename = filename
        base = filename.split('_seed')[0] if '_seed' in filename else filename
        parts = base.split('-')

        # 工具函数：更鲁棒的数字判定（支持 1e-3 等）
        def _is_number(s: str) -> bool:
            return bool(re.match(r"^\d*\.?\d+(?:e[+-]?\d+)?$", s))

        # 评估策略前缀
        eval_tokens = {'inter-session', 'inter-subject', 'cross-session', 'cross-subject'}
        evaluation = 'unknown'
        eval_offset = 0
        if parts and parts[0] in eval_tokens:
            evaluation = parts[0]
            eval_offset = 1

        sampling_set = {'lsm', 'olm', 'ecm', 'lecm'}
        optimizers = {'AMSGRAD', 'Adam', 'AdamW', 'SGD', 'RMSprop', 'LBFGS'}
        toggles = {'LPon', 'LPoff', 'LOGMon', 'LOGMoff'}

        # seed
        seed = 'unknown'
        m = re.search(r"_seed(\d+)", raw_filename)
        if m:
            seed = m.group(1)

        # net/loss lr：优先使用 nLR/lLR，若无则回退到首两个数字
        net_lr = 'unknown'
        loss_lr = 'unknown'
        for p in parts:
            if p.startswith('nLR'):
                net_lr = p[3:]
            elif p.startswith('lLR'):
                loss_lr = p[3:]

        base_idx = eval_offset
        if net_lr == 'unknown' and len(parts) > base_idx and _is_number(parts[base_idx]):
            net_lr = parts[base_idx]
        if loss_lr == 'unknown' and len(parts) > base_idx + 1 and _is_number(parts[base_idx + 1]):
            loss_lr = parts[base_idx + 1]

        # 优化器与 weight_decay（若文件名未包含则置 unknown）
        optimizer = 'unknown'
        weight_decay = 'unknown'
        opt_idx = -1
        for i, p in enumerate(parts):
            if p in optimizers:
                optimizer = p
                opt_idx = i
                if i + 1 < len(parts) and _is_number(parts[i + 1]):
                    weight_decay = parts[i + 1]
                break

        # n_proj 与架构：从出现 'P' 的位置作为界标；架构从包含网络名的段开始到 'P' 之前
        p_idx = None
        for i, p in enumerate(parts):
            if p.startswith('P'):
                p_idx = i
                break

        # 架构起点：优先匹配包含 TSMNet/SPDNet/TSMNet+ 等标识符
        arch_start = None
        for i, p in enumerate(parts):
            if ('TSMNet' in p) or ('SPDNet' in p) or ('TSMNet+' in p) or ('SPDDSMBN' in p):
                arch_start = i
                break

        architect = 'unknown'
        if arch_start is not None:
            end = p_idx if p_idx is not None else len(parts)
            arch_tokens = [t.split('_seed')[0] for t in parts[arch_start:end]]
            if arch_tokens:
                architect = '-'.join(arch_tokens)
        else:
            # 兼容简化命名：若未命中网络标识，尝试使用第三段作为架构
            arch_idx = 2 + eval_offset
            if len(parts) > arch_idx:
                architect = parts[arch_idx]

        n_proj = 'unknown'
        if p_idx is not None:
            token = parts[p_idx]
            n_proj = token[1:] if len(token) > 1 else 'unknown'

        # sampling：一般在 P 之后一段
        sampling = 'unknown'
        if p_idx is not None and p_idx + 1 < len(parts):
            nxt = parts[p_idx + 1]
            if nxt in sampling_set:
                sampling = nxt

        # LP/LOGM 开关标记（文件名显式标记优先）
        lp = 'unknown'
        logm = 'unknown'
        for p in parts:
            if p in toggles:
                if p.startswith('LP'):
                    lp = 'on' if p == 'LPon' else 'off'
                elif p.startswith('LOGM'):
                    logm = 'on' if p == 'LOGMon' else 'off'

        # lambda/gamma（用于推断 LP/LOGM 开关）
        lamda = 'unknown'
        gamma = 'unknown'
        for p in parts:
            if p.startswith('L') and _is_number(p[1:]):
                lamda = p[1:]
            elif p.startswith('G') and _is_number(p[1:]):
                gamma = p[1:]

        # 若 LP/LOGM 未显式标注，则根据 L/G 数值判断（>0 => on；=0 => off）
        try:
            if lp == 'unknown' and lamda != 'unknown':
                lp = 'on' if float(lamda) > 0 else 'off'
        except Exception:
            pass
        try:
            if logm == 'unknown' and gamma != 'unknown':
                logm = 'on' if float(gamma) > 0 else 'off'
        except Exception:
            pass

        # init λ：扫描任何以 'init' 开头的 token（格式 init<l1>_<l2>）
        init_l1 = 'unknown'
        init_l2 = 'unknown'
        for p in parts:
            if p.startswith('init'):
                payload = p[4:]
                if '_' in payload:
                    l1, l2 = payload.split('_', 1)
                    init_l1 = l1 if l1 else 'unknown'
                    init_l2 = l2 if l2 else 'unknown'
                else:
                    init_l1 = payload if payload else 'unknown'
                break

        # power：优先识别 'Pow'；若未出现则保持 unknown（不强行猜测）
        power = 'unknown'
        for p in parts[::-1]:
            if p.startswith('Pow'):
                p0 = p.split('_')[0]
                power = p0[3:] if len(p0) > 3 else 'unknown'
                break

        # 可能在 .pt 中保存 subject/session；文件名默认不解析
        subject = 'unknown'
        session = 'unknown'

        return {
            'seed': seed,
            'net_lr': net_lr,
            'loss_lr': loss_lr,
            'optimizer': optimizer,
            'weight_decay': weight_decay,
            'architect': architect,
            'n_proj': n_proj,
            'sampling': sampling,
            'lp': lp,
            'logm': logm,
            'lamda': lamda,
            'gamma': gamma,
            'init_l1': init_l1,
            'init_l2': init_l2,
            'power': power,
            'evaluation': evaluation,
            'subject': subject,
            'session': session,
        }
    except Exception as e:
        print(f"解析文件名 {filename} 时出错: {e}")
        raise


def get_group_key(params, group_by=['evaluation', 'net_lr', 'loss_lr', 'optimizer', 'weight_decay',
                                     'architect', 'n_proj', 'sampling', 'lp', 'logm', 'lamda', 'gamma', 'init_l1', 'init_l2', 'power']):
    """
    根据指定参数生成分组键
    
    Parameters
    ----------
    params : dict
        参数字典
    group_by : list
        用于分组的参数列表
        
    Returns
    -------
    str
        分组键
    """
    key_parts = []
    for param in group_by:
        if param in params:
            key_parts.append(f"{param}={params[param]}")
    return '|'.join(key_parts)


def calculate_statistics(results_dir='/root/SPDMLR-main/torch_results', include_seeds=None):
    """
    计算不同参数组合下的统计信息
    
    Parameters
    ----------
    results_dir : str
        结果文件目录
        
    Returns
    -------
    dict
        包含各组统计信息的字典
    """
    # 获取所有文件
    all_files = os.listdir(results_dir)
    files = [f for f in all_files if os.path.isfile(os.path.join(results_dir, f))]
    
    # 按组分类结果
    groups = defaultdict(list)
    
    print(f"目录中共有 {len(all_files)} 个项目")
    print(f"其中文件数量: {len(files)}\n")
    
    parse_errors = []
    load_errors = []
    
    # 解析每个文件
    for filename in files:
        try:
            # 解析文件名
            params = parse_filename(filename)
            # 如提供 seeds，仅保留这些 seed
            if include_seeds is not None and params.get('seed') not in include_seeds:
                continue
            
            # 加载结果
            file_path = os.path.join(results_dir, filename)
            
            try:
                data = torch.load(file_path, map_location='cpu')
            except Exception as e:
                load_errors.append((filename, str(e)))
                continue
            
            # 将数据中的评估策略/被试/会话信息写回参数，用于分组
            eval_from_data = data.get('evaluation', None)
            subject_from_data = data.get('subject', None)
            session_from_data = data.get('session', None)
            lp_from_data = data.get('lp', None)
            logm_from_data = data.get('logm', None)
            init_l1_from_data = data.get('init_l1', None)
            init_l2_from_data = data.get('init_l2', None)
            power_from_data = data.get('power', None)
            if eval_from_data is not None:
                params['evaluation'] = str(eval_from_data)
            if subject_from_data is not None:
                params['subject'] = str(subject_from_data)
            if session_from_data is not None:
                params['session'] = str(session_from_data)
            if lp_from_data is not None:
                params['lp'] = 'on' if str(lp_from_data).lower() in ('on', 'true', '1') else (
                    'off' if str(lp_from_data).lower() in ('off', 'false', '0') else str(lp_from_data)
                )
            if logm_from_data is not None:
                params['logm'] = 'on' if str(logm_from_data).lower() in ('on', 'true', '1') else (
                    'off' if str(logm_from_data).lower() in ('off', 'false', '0') else str(logm_from_data)
                )
            if init_l1_from_data is not None:
                params['init_l1'] = str(init_l1_from_data)
            if init_l2_from_data is not None:
                params['init_l2'] = str(init_l2_from_data)
            if power_from_data is not None:
                params['power'] = str(power_from_data)

            # 最终后备：若文件名解析与 payload 都未给出 lp/logm，则根据 L/G 数值推断
            try:
                if params.get('lp', 'unknown') == 'unknown' and params.get('lamda', 'unknown') != 'unknown':
                    params['lp'] = 'on' if float(params['lamda']) > 0 else 'off'
            except Exception:
                pass
            try:
                if params.get('logm', 'unknown') == 'unknown' and params.get('gamma', 'unknown') != 'unknown':
                    params['logm'] = 'on' if float(params['gamma']) > 0 else 'off'
            except Exception:
                pass

            # 生成分组键 (排除 seed，加入 evaluation)
            group_key = get_group_key(params, group_by=['evaluation', 'net_lr', 'loss_lr', 'optimizer', 
                                                         'weight_decay', 'architect', 'n_proj', 
                                                         'sampling', 'lp', 'logm', 'lamda', 'gamma', 'init_l1', 'init_l2', 'power'])

            # 提取最终准确率
            final_acc = data.get('final_acc', None)
            best_acc = data.get('best_acc', None)

            # 如果没有找到 final_acc，尝试从其他可能的键获取
            if final_acc is None:
                # 尝试其他可能的键名（兼容 EEG 结果格式）
                for key in ['final_test_acc', 'test_acc', 'acc', 'score_tst_mean', 'score_tst']:
                    if key in data:
                        final_acc = data[key]
                        break
            
            if final_acc is not None:
                groups[group_key].append({
                    'seed': params['seed'],
                    'final_acc': final_acc,
                    'best_acc': best_acc,
                    'filename': filename,
                    'params': params
                })
            else:
                print(f"警告: 文件 {filename} 中未找到准确率数据")
                print(f"  可用的键: {list(data.keys())}")
                
        except Exception as e:
            parse_errors.append((filename, str(e)))
            continue
    
    # 打印错误信息
    if parse_errors:
        print(f"\n解析错误 ({len(parse_errors)} 个文件):")
        for fname, error in parse_errors[:5]:  # 只显示前5个
            print(f"  - {fname}: {error}")
        if len(parse_errors) > 5:
            print(f"  ... 还有 {len(parse_errors) - 5} 个文件解析失败")
    
    if load_errors:
        print(f"\n加载错误 ({len(load_errors)} 个文件):")
        for fname, error in load_errors[:5]:
            print(f"  - {fname}: {error}")
        if len(load_errors) > 5:
            print(f"  ... 还有 {len(load_errors) - 5} 个文件加载失败")
    
    print(f"\n成功处理的文件分为 {len(groups)} 个参数组合")
    
    # 计算每组的统计信息
    statistics = {}
    
    for group_key, results in groups.items():
        if len(results) == 0:
            continue
        
        final_accs = [r['final_acc'] for r in results]
        best_accs = [r['best_acc'] for r in results if r['best_acc'] is not None]
        
        statistics[group_key] = {
            'params': results[0]['params'],  # 组的参数
            'n_runs': len(results),  # 运行次数
            'seeds': [r['seed'] for r in results],  # 所有seed
            'final_acc_mean': np.mean(final_accs),
            'final_acc_std': np.std(final_accs),
            'final_acc_min': np.min(final_accs),
            'final_acc_max': np.max(final_accs),
            'final_acc_all': final_accs,
            'best_acc_mean': np.mean(best_accs) if best_accs else None,
            'best_acc_std': np.std(best_accs) if best_accs else None,
            'best_acc_min': np.min(best_accs) if best_accs else None,
            'best_acc_max': np.max(best_accs) if best_accs else None,
            'best_acc_all': best_accs
        }
    
    return statistics


def _aggregate_statistics_by_init(statistics):
    """将现有细粒度统计按 init 聚合，返回新的统计字典。"""
    agg = defaultdict(lambda: {
        'params': {'init_l1': 'unknown', 'init_l2': 'unknown'},
        'n_runs': 0,
        'seeds': [],
        'final_acc_all': [],
        'best_acc_all': []
    })

    for _, stats in statistics.items():
        init_l1 = stats['params'].get('init_l1', 'unknown')
        init_l2 = stats['params'].get('init_l2', 'unknown')
        key = f"init_l1={init_l1}|init_l2={init_l2}"

        agg[key]['params'] = {'init_l1': init_l1, 'init_l2': init_l2}
        agg[key]['n_runs'] += stats['n_runs']
        agg[key]['seeds'].extend(stats.get('seeds', []))
        agg[key]['final_acc_all'].extend(stats.get('final_acc_all', []))
        if stats.get('best_acc_all'):
            agg[key]['best_acc_all'].extend(stats.get('best_acc_all', []))

    # 完成均值与方差等派生量的计算
    aggregated = {}
    for key, s in agg.items():
        finals = np.array(s['final_acc_all'], dtype=float) if len(s['final_acc_all']) > 0 else np.array([], dtype=float)
        bests = np.array(s['best_acc_all'], dtype=float) if len(s['best_acc_all']) > 0 else np.array([], dtype=float)

        aggregated[key] = {
            'params': s['params'],
            'n_runs': s['n_runs'],
            'seeds': s['seeds'],
            'final_acc_mean': float(np.mean(finals)) if finals.size > 0 else None,
            'final_acc_std': float(np.std(finals)) if finals.size > 0 else None,
            'final_acc_min': float(np.min(finals)) if finals.size > 0 else None,
            'final_acc_max': float(np.max(finals)) if finals.size > 0 else None,
            'final_acc_all': s['final_acc_all'],
            'best_acc_mean': float(np.mean(bests)) if bests.size > 0 else None,
            'best_acc_std': float(np.std(bests)) if bests.size > 0 else None,
            'best_acc_min': float(np.min(bests)) if bests.size > 0 else None,
            'best_acc_max': float(np.max(bests)) if bests.size > 0 else None,
            'best_acc_all': s['best_acc_all']
        }
    return aggregated


def print_statistics(statistics, primary_group='full'):
    """
    打印统计结果
    
    Parameters
    ----------
    statistics : dict
        统计信息字典
    primary_group : str
        顶部主表的分组方式：'full' 使用细粒度组合；'init' 按 init 聚合
    """
    print("\n" + "=" * 180)
    header = "Group" if primary_group == 'full' else "Init Group"
    print(f"{header:<120} {'N':<5} {'Final Acc (Mean±Std)':<30} {'Best Acc (Mean±Std)':<30}")
    print("=" * 180)

    # 选择数据来源：细粒度或按 init 聚合
    if primary_group == 'init':
        source = _aggregate_statistics_by_init(statistics)
    else:
        source = statistics

    # 按 final_acc_mean 降序排序（None 置底）
    def _sort_key(item):
        v = item[1].get('final_acc_mean')
        return (v is not None, v if v is not None else -1e9)

    sorted_groups = sorted(source.items(), key=_sort_key, reverse=True)

    for group_key, stats in sorted_groups:
        if stats['final_acc_mean'] is not None:
            final_acc_str = f"{stats['final_acc_mean']:.4f}±{stats['final_acc_std']:.4f} [{stats['final_acc_min']:.4f}-{stats['final_acc_max']:.4f}]"
        else:
            final_acc_str = "N/A"

        if stats['best_acc_mean'] is not None:
            best_acc_str = f"{stats['best_acc_mean']:.4f}±{stats['best_acc_std']:.4f} [{stats['best_acc_min']:.4f}-{stats['best_acc_max']:.4f}]"
        else:
            best_acc_str = "N/A"

        print(f"{group_key:<120} {stats['n_runs']:<5} {final_acc_str:<30} {best_acc_str:<30}")

    print("=" * 180)


def print_grouped_by_n_proj(statistics):
    """
    按 n_proj 分组打印统计结果，方便比较不同 n_proj 的效果
    
    Parameters
    ----------
    statistics : dict
        统计信息字典
    """
    # 按 n_proj 分组
    n_proj_groups = defaultdict(list)
    
    for group_key, stats in statistics.items():
        n_proj = stats['params']['n_proj']
        n_proj_groups[n_proj].append((group_key, stats))
    
    print("\n" + "=" * 180)
    print("按 n_proj 分组的结果:")
    print("=" * 180)
    
    # 按 n_proj 数值排序
    sorted_n_projs = sorted(n_proj_groups.keys(), key=lambda x: int(x) if x.isdigit() else 0)
    
    for n_proj in sorted_n_projs:
        print(f"\n**n_proj = {n_proj}**")
        print("-" * 180)
        print(f"{'Configuration':<110} {'N':<5} {'Final Acc (Mean±Std)':<30} {'Best Acc (Mean±Std)':<30}")
        print("-" * 180)
        
        # 按 final_acc_mean 降序排序
        sorted_configs = sorted(n_proj_groups[n_proj], key=lambda x: x[1]['final_acc_mean'], reverse=True)
        
        for group_key, stats in sorted_configs:
            # 移除 n_proj 信息，使配置更简洁
            config = group_key.replace(f'|n_proj={n_proj}', '')
            
            final_acc_str = f"{stats['final_acc_mean']:.4f}±{stats['final_acc_std']:.4f}"
            
            if stats['best_acc_mean'] is not None:
                best_acc_str = f"{stats['best_acc_mean']:.4f}±{stats['best_acc_std']:.4f}"
            else:
                best_acc_str = "N/A"
            
            print(f"{config:<110} {stats['n_runs']:<5} {final_acc_str:<30} {best_acc_str:<30}")


def print_grouped_by_sampling(statistics):
    """
    按 sampling (metric) 分组打印统计结果
    
    Parameters
    ----------
    statistics : dict
        统计信息字典
    """
    # 按 sampling 分组
    sampling_groups = defaultdict(list)
    
    for group_key, stats in statistics.items():
        sampling = stats['params']['sampling']
        sampling_groups[sampling].append((group_key, stats))
    
    print("\n" + "=" * 180)
    print("按 sampling (metric) 分组的结果:")
    print("=" * 180)
    
    for sampling in sorted(sampling_groups.keys()):
        print(f"\n**sampling = {sampling}**")
        print("-" * 180)
        print(f"{'Configuration':<110} {'N':<5} {'Final Acc (Mean±Std)':<30} {'Best Acc (Mean±Std)':<30}")
        print("-" * 180)
        
        # 按 final_acc_mean 降序排序
        sorted_configs = sorted(sampling_groups[sampling], key=lambda x: x[1]['final_acc_mean'], reverse=True)
        
        for group_key, stats in sorted_configs:
            # 移除 sampling 信息
            config = group_key.replace(f'|sampling={sampling}', '')
            
            final_acc_str = f"{stats['final_acc_mean']:.4f}±{stats['final_acc_std']:.4f}"
            
            if stats['best_acc_mean'] is not None:
                best_acc_str = f"{stats['best_acc_mean']:.4f}±{stats['best_acc_std']:.4f}"
            else:
                best_acc_str = "N/A"
            
            print(f"{config:<110} {stats['n_runs']:<5} {final_acc_str:<30} {best_acc_str:<30}")


def print_grouped_by_lp_logm(statistics):
    """
    按 LP/LOGM 开关分组打印统计结果
    
    Parameters
    ----------
    statistics : dict
        统计信息字典
    """
    lp_logm_groups = defaultdict(list)
    for group_key, stats in statistics.items():
        lp = stats['params'].get('lp', 'unknown')
        logm = stats['params'].get('logm', 'unknown')
        lp_logm_groups[(lp, logm)].append((group_key, stats))

    print("\n" + "=" * 180)
    print("按 LP/LOGM 分组的结果:")
    print("=" * 180)

    order = {'on': 2, 'off': 1, 'unknown': 0}
    keys_sorted = sorted(lp_logm_groups.keys(), key=lambda x: (order.get(x[0], 0), order.get(x[1], 0)))

    for lp, logm in keys_sorted:
        print(f"\n**LP = {lp} | LOGM = {logm}**")
        print("-" * 180)
        print(f"{'Configuration':<110} {'N':<5} {'Final Acc (Mean±Std)':<30} {'Best Acc (Mean±Std)':<30}")
        print("-" * 180)

        sorted_configs = sorted(lp_logm_groups[(lp, logm)], key=lambda x: x[1]['final_acc_mean'], reverse=True)
        for group_key, stats in sorted_configs:
            cfg_display = group_key.replace(f'|lp={lp}', '').replace(f'|logm={logm}', '')
            final_acc_str = f"{stats['final_acc_mean']:.4f}±{stats['final_acc_std']:.4f}"
            if stats['best_acc_mean'] is not None:
                best_acc_str = f"{stats['best_acc_mean']:.4f}±{stats['best_acc_std']:.4f}"
            else:
                best_acc_str = "N/A"
            print(f"{cfg_display:<110} {stats['n_runs']:<5} {final_acc_str:<30} {best_acc_str:<30}")


def print_grouped_by_init(statistics):
    """
    按初始 λ 值分组打印统计结果
    """
    init_groups = defaultdict(list)
    for group_key, stats in statistics.items():
        init_l1 = stats['params'].get('init_l1', 'unknown')
        init_l2 = stats['params'].get('init_l2', 'unknown')
        init_groups[(init_l1, init_l2)].append((group_key, stats))

    print("\n" + "=" * 180)
    print("按 init λ 分组的结果:")
    print("=" * 180)

    def _num_or_str(x):
        try:
            return float(x)
        except:
            return -1e9

    keys_sorted = sorted(init_groups.keys(), key=lambda x: (_num_or_str(x[0]), _num_or_str(x[1])))

    for init_l1, init_l2 in keys_sorted:
        print(f"\n**init_l1 = {init_l1} | init_l2 = {init_l2}**")
        print("-" * 180)
        print(f"{'Configuration':<110} {'N':<5} {'Final Acc (Mean±Std)':<30} {'Best Acc (Mean±Std)':<30}")
        print("-" * 180)

        sorted_configs = sorted(init_groups[(init_l1, init_l2)], key=lambda x: x[1]['final_acc_mean'], reverse=True)
        for group_key, stats in sorted_configs:
            cfg_display = group_key.replace(f'|init_l1={init_l1}', '').replace(f'|init_l2={init_l2}', '')
            final_acc_str = f"{stats['final_acc_mean']:.4f}±{stats['final_acc_std']:.4f}"
            if stats['best_acc_mean'] is not None:
                best_acc_str = f"{stats['best_acc_mean']:.4f}±{stats['best_acc_std']:.4f}"
            else:
                best_acc_str = "N/A"
            print(f"{cfg_display:<110} {stats['n_runs']:<5} {final_acc_str:<30} {best_acc_str:<30}")


def save_statistics_to_file(statistics, output_file='statistics_summary.txt', primary_group='full'):
    """
    将统计结果保存到文件
    
    Parameters
    ----------
    statistics : dict
        统计信息字典
    output_file : str
        输出文件路径
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 180 + "\n")
        header = "Group" if primary_group == 'full' else "Init Group"
        f.write(f"{header:<120} {'N':<5} {'Final Acc (Mean±Std)':<30} {'Best Acc (Mean±Std)':<30}\n")
        f.write("=" * 180 + "\n")

        # 选择数据来源：细粒度或按 init 聚合
        if primary_group == 'init':
            source = _aggregate_statistics_by_init(statistics)
        else:
            source = statistics

        # 按 final_acc_mean 降序排序（None 置底）
        def _sort_key(item):
            v = item[1].get('final_acc_mean')
            return (v is not None, v if v is not None else -1e9)

        sorted_groups = sorted(source.items(), key=_sort_key, reverse=True)

        for group_key, stats in sorted_groups:
            if stats['final_acc_mean'] is not None:
                final_acc_str = f"{stats['final_acc_mean']:.4f}±{stats['final_acc_std']:.4f} [{stats['final_acc_min']:.4f}-{stats['final_acc_max']:.4f}]"
            else:
                final_acc_str = "N/A"

            if stats['best_acc_mean'] is not None:
                best_acc_str = f"{stats['best_acc_mean']:.4f}±{stats['best_acc_std']:.4f} [{stats['best_acc_min']:.4f}-{stats['best_acc_max']:.4f}]"
            else:
                best_acc_str = "N/A"

            f.write(f"{group_key:<120} {stats['n_runs']:<5} {final_acc_str:<30} {best_acc_str:<30}\n")

            # 主表详细信息
            f.write(f"  Seeds: {', '.join(stats['seeds'])}\n")
            if stats.get('final_acc_all'):
                f.write(f"  Final accuracies: {[f'{acc:.4f}' for acc in stats['final_acc_all']]}\n")
            if stats.get('best_acc_all'):
                f.write(f"  Best accuracies: {[f'{acc:.4f}' for acc in stats['best_acc_all']]}\n")
            f.write("\n")

        f.write("=" * 180 + "\n")
        
        # 添加按 n_proj 分组的结果
        f.write("\n\n")
        f.write("=" * 180 + "\n")
        f.write("按 n_proj 分组的结果:\n")
        f.write("=" * 180 + "\n")
        
        # 按 n_proj 分组
        n_proj_groups = defaultdict(list)
        for group_key, stats in statistics.items():
            n_proj = stats['params']['n_proj']
            n_proj_groups[n_proj].append((group_key, stats))
        
        # 按 n_proj 数值排序
        sorted_n_projs = sorted(n_proj_groups.keys(), key=lambda x: int(x) if x.isdigit() else 0)
        
        for n_proj in sorted_n_projs:
            f.write(f"\n**n_proj = {n_proj}**\n")
            f.write("-" * 180 + "\n")
            f.write(f"{'Configuration':<110} {'N':<5} {'Final Acc (Mean±Std)':<30} {'Best Acc (Mean±Std)':<30}\n")
            f.write("-" * 180 + "\n")
            
            sorted_configs = sorted(n_proj_groups[n_proj], key=lambda x: x[1]['final_acc_mean'], reverse=True)
            
            for group_key, stats in sorted_configs:
                config = group_key.replace(f'|n_proj={n_proj}', '')
                
                final_acc_str = f"{stats['final_acc_mean']:.4f}±{stats['final_acc_std']:.4f}"
                
                if stats['best_acc_mean'] is not None:
                    best_acc_str = f"{stats['best_acc_mean']:.4f}±{stats['best_acc_std']:.4f}"
                else:
                    best_acc_str = "N/A"
                
                f.write(f"{config:<110} {stats['n_runs']:<5} {final_acc_str:<30} {best_acc_str:<30}\n")
        
        # 添加按 sampling 分组的结果
        f.write("\n\n")
        f.write("=" * 180 + "\n")
        f.write("按 sampling (metric) 分组的结果:\n")
        f.write("=" * 180 + "\n")
        
        sampling_groups = defaultdict(list)
        for group_key, stats in statistics.items():
            sampling = stats['params']['sampling']
            sampling_groups[sampling].append((group_key, stats))
        
        for sampling in sorted(sampling_groups.keys()):
            f.write(f"\n**sampling = {sampling}**\n")
            f.write("-" * 180 + "\n")
            f.write(f"{'Configuration':<110} {'N':<5} {'Final Acc (Mean±Std)':<30} {'Best Acc (Mean±Std)':<30}\n")
            f.write("-" * 180 + "\n")
            
            sorted_configs = sorted(sampling_groups[sampling], key=lambda x: x[1]['final_acc_mean'], reverse=True)
            
            for group_key, stats in sorted_configs:
                config = group_key.replace(f'|sampling={sampling}', '')
                
                final_acc_str = f"{stats['final_acc_mean']:.4f}±{stats['final_acc_std']:.4f}"
                
                if stats['best_acc_mean'] is not None:
                    best_acc_str = f"{stats['best_acc_mean']:.4f}±{stats['best_acc_std']:.4f}"
                else:
                    best_acc_str = "N/A"
                
                f.write(f"{config:<110} {stats['n_runs']:<5} {final_acc_str:<30} {best_acc_str:<30}\n")

        # 添加按 init λ 分组的结果
        f.write("\n\n")
        f.write("=" * 180 + "\n")
        f.write("按 init λ 分组的结果:\n")
        f.write("=" * 180 + "\n")

        init_groups = defaultdict(list)
        for group_key, stats in statistics.items():
            init_l1 = stats['params'].get('init_l1', 'unknown')
            init_l2 = stats['params'].get('init_l2', 'unknown')
            init_groups[(init_l1, init_l2)].append((group_key, stats))

        def _num_or_str(x):
            try:
                return float(x)
            except:
                return -1e9

        keys_sorted = sorted(init_groups.keys(), key=lambda x: (_num_or_str(x[0]), _num_or_str(x[1])))

        for init_l1, init_l2 in keys_sorted:
            f.write(f"\n**init_l1 = {init_l1} | init_l2 = {init_l2}**\n")
            f.write("-" * 180 + "\n")
            f.write(f"{'Configuration':<110} {'N':<5} {'Final Acc (Mean±Std)':<30} {'Best Acc (Mean±Std)':<30}\n")
            f.write("-" * 180 + "\n")
            sorted_configs = sorted(init_groups[(init_l1, init_l2)], key=lambda x: x[1]['final_acc_mean'], reverse=True)
            for group_key, stats in sorted_configs:
                cfg_display = group_key.replace(f'|init_l1={init_l1}', '').replace(f'|init_l2={init_l2}', '')
                final_acc_str = f"{stats['final_acc_mean']:.4f}±{stats['final_acc_std']:.4f}"
                if stats['best_acc_mean'] is not None:
                    best_acc_str = f"{stats['best_acc_mean']:.4f}±{stats['best_acc_std']:.4f}"
                else:
                    best_acc_str = "N/A"
                f.write(f"{cfg_display:<110} {stats['n_runs']:<5} {final_acc_str:<30} {best_acc_str:<30}\n")
    
    print(f"\n统计结果已保存到: {output_file}")


def analyze_parameter_effects(statistics):
    """
    分析各参数对性能的影响
    
    Parameters
    ----------
    statistics : dict
        统计信息字典
    """
    print("\n" + "=" * 100)
    print("参数影响分析:")
    print("=" * 100)
    
    # 分析 n_proj 的影响
    print(f"\n1. n_proj 的影响:")
    n_proj_performance = defaultdict(list)
    for group_key, stats in statistics.items():
        n_proj = stats['params']['n_proj']
        n_proj_performance[n_proj].append(stats['final_acc_mean'])
    
    sorted_n_projs = sorted(n_proj_performance.keys(), key=lambda x: int(x) if x.isdigit() else 0)
    for n_proj in sorted_n_projs:
        avg_acc = np.mean(n_proj_performance[n_proj])
        std_acc = np.std(n_proj_performance[n_proj])
        print(f"   n_proj={n_proj:>3}: {avg_acc:.4f}±{std_acc:.4f} (基于 {len(n_proj_performance[n_proj])} 个配置)")
    
    # 分析 sampling 的影响
    print(f"\n2. sampling (metric) 的影响:")
    sampling_performance = defaultdict(list)
    for group_key, stats in statistics.items():
        sampling = stats['params']['sampling']
        sampling_performance[sampling].append(stats['final_acc_mean'])
    
    for sampling in sorted(sampling_performance.keys()):
        avg_acc = np.mean(sampling_performance[sampling])
        std_acc = np.std(sampling_performance[sampling])
        print(f"   sampling={sampling}: {avg_acc:.4f}±{std_acc:.4f} (基于 {len(sampling_performance[sampling])} 个配置)")
    
    # 分析 learning rate 的影响
    print(f"\n3. learning rate 的影响:")
    lr_performance = defaultdict(list)
    for group_key, stats in statistics.items():
        lr_key = f"net={stats['params']['net_lr']}, loss={stats['params']['loss_lr']}"
        lr_performance[lr_key].append(stats['final_acc_mean'])
    
    sorted_lrs = sorted(lr_performance.items(), key=lambda x: np.mean(x[1]), reverse=True)
    for lr_key, accs in sorted_lrs[:10]:  # 只显示前10个
        avg_acc = np.mean(accs)
        std_acc = np.std(accs)
        print(f"   {lr_key}: {avg_acc:.4f}±{std_acc:.4f} (基于 {len(accs)} 个配置)")
    
    # 分析 lambda 的影响
    print(f"\n4. lambda (lamda) 的影响:")
    lamda_performance = defaultdict(list)
    for group_key, stats in statistics.items():
        lamda = stats['params']['lamda']
        lamda_performance[lamda].append(stats['final_acc_mean'])
    
    sorted_lamdas = sorted(lamda_performance.items(), key=lambda x: float(x[0]) if x[0].replace('.', '').isdigit() else 0)
    for lamda, accs in sorted_lamdas:
        avg_acc = np.mean(accs)
        std_acc = np.std(accs)
        print(f"   lamda={lamda}: {avg_acc:.4f}±{std_acc:.4f} (基于 {len(accs)} 个配置)")
    
    # 分析 gamma 的影响
    print(f"\n5. gamma 的影响:")
    gamma_performance = defaultdict(list)
    for group_key, stats in statistics.items():
        gamma = stats['params']['gamma']
        gamma_performance[gamma].append(stats['final_acc_mean'])
    
    sorted_gammas = sorted(gamma_performance.items(), key=lambda x: float(x[0]) if x[0].replace('.', '').replace('-', '').isdigit() else 0)
    for gamma, accs in sorted_gammas:
        avg_acc = np.mean(accs)
        std_acc = np.std(accs)
        print(f"   gamma={gamma}: {avg_acc:.4f}±{std_acc:.4f} (基于 {len(accs)} 个配置)")


def main():
    """主函数"""
    # 解析命令行参数，支持手动指定输入/输出路径
    parser = argparse.ArgumentParser(description="统计多 seed 结果并支持自定义路径")
    parser.add_argument("--results_dir", "-r", type=str, default=None, help="结果文件目录，不指定则默认为脚本同级的 torch_results")
    parser.add_argument("--output_file", "-o", type=str, default=None, help="保存统计结果的文件路径；不指定则保存到脚本同级 statistics_summary.txt")
    parser.add_argument("--seeds", "-s", nargs="+", help="仅统计指定的 seeds（空格分隔，示例：-s 12 34 56）")
    parser.add_argument("--primary_group", type=str, default="full", choices=["full", "init"], help="主表的分组方式：full=按完整配置分组；init=按初始 λ 分组")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.abspath(os.path.expanduser(args.results_dir)) if args.results_dir else os.path.join(script_dir, "torch_results")
    output_file = os.path.abspath(os.path.expanduser(args.output_file)) if args.output_file else os.path.join(script_dir, "statistics_summary.txt")
    include_seeds = set(item for s in args.seeds for item in s.split(',')) if args.seeds else None

    print("开始计算统计信息...\n")

    if not os.path.isdir(results_dir):
        print(f"错误: 结果目录不存在: {results_dir}")
        return None

    # 计算统计信息
    statistics = calculate_statistics(results_dir=results_dir, include_seeds=include_seeds)

    if len(statistics) == 0:
        print("\n错误: 未找到有效的统计数据!")
        return None

    print(f"\n共找到 {len(statistics)} 个参数组合\n")

    # 打印统计结果（主表）
    print_statistics(statistics, primary_group=args.primary_group)

    # 按 n_proj 分组打印
    print_grouped_by_n_proj(statistics)

    # 按 sampling 分组打印
    print_grouped_by_sampling(statistics)

    # 按 LP/LOGM 分组打印
    print_grouped_by_lp_logm(statistics)

    # 按 init λ 分组打印
    print_grouped_by_init(statistics)

    # 参数影响分析
    analyze_parameter_effects(statistics)

    # 保存到文件（手动输出路径）
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    save_statistics_to_file(statistics, output_file=output_file, primary_group=args.primary_group)

    # 打印最佳组合
    print("\n" + "=" * 100)
    print("最佳组合:")
    print("=" * 100)

    # 找出表现最好的组合
    best_group = max(statistics.items(), key=lambda x: x[1]['final_acc_mean'])
    print(f"\n🏆 最佳组合 (基于 Final Acc Mean):")
    print(f"   参数: {best_group[0]}")
    print(f"   Final Acc: {best_group[1]['final_acc_mean']:.4f}±{best_group[1]['final_acc_std']:.4f}")
    print(f"   范围: [{best_group[1]['final_acc_min']:.4f}, {best_group[1]['final_acc_max']:.4f}]")
    print(f"   运行次数: {best_group[1]['n_runs']}")
    print(f"   Seeds: {', '.join(best_group[1]['seeds'])}")
    if best_group[1]['best_acc_mean'] is not None:
        print(f"   Best Acc: {best_group[1]['best_acc_mean']:.4f}±{best_group[1]['best_acc_std']:.4f}")

    return statistics


if __name__ == "__main__":
    statistics = main()