"""
检测ManiSkill可用的环境任务
"""
import gymnasium as gym

print("正在加载ManiSkill环境...")
try:
    import mani_skill.envs
    print("✓ ManiSkill导入成功\n")
except ImportError as e:
    print(f"✗ ManiSkill导入失败: {e}")
    print("请先安装: pip install mani-skill")
    exit(1)

# 获取所有注册的环境
all_envs = list(gym.envs.registry.keys())

# 筛选ManiSkill相关的环境
maniskill_envs = [env for env in all_envs if any(keyword in env.lower() for keyword in 
                  ['push', 'pick', 'place', 'open', 'close', 'drawer', 'door', 'cabinet', 'peg', 'stack'])]

print(f"找到 {len(maniskill_envs)} 个相关任务:\n")
print("="*60)

# 按类别分组
categories = {
    'Push类': [],
    'Pick/Place类': [],
    'Open/Close类': [],
    '其他': []
}

for env in sorted(maniskill_envs):
    if 'push' in env.lower():
        categories['Push类'].append(env)
    elif any(kw in env.lower() for kw in ['pick', 'place', 'stack', 'peg']):
        categories['Pick/Place类'].append(env)
    elif any(kw in env.lower() for kw in ['open', 'close', 'drawer', 'door', 'cabinet']):
        categories['Open/Close类'].append(env)
    else:
        categories['其他'].append(env)

for category, envs in categories.items():
    if envs:
        print(f"\n{category} ({len(envs)} 个):")
        for env in envs:
            print(f"  - {env}")

print("\n" + "="*60)
print("\n推荐使用的任务（根据你的需求）:")
print("\nPush类:")
print("  PushCube-v1  (如果存在)")
print("  或尝试: PushCube-v0")

print("\nPick类:")
print("  PickCube-v1  (如果存在)")
print("  或尝试: PickCube-v0")

print("\nOpen类:")
print("  OpenCabinetDrawer-v1  (如果存在)")
print("  或尝试: OpenCabinetDrawer-v0")

print("\n" + "="*60)
print("\n测试某个环境是否可用:")
print("  python -c \"import gymnasium as gym; import mani_skill.envs; env = gym.make('PushCube-v1'); print('✓ 可用')\"")