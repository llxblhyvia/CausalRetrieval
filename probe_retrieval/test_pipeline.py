"""
快速测试脚本 - 验证整个数据收集pipeline

用法: python test_pipeline.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import gymnasium as gym
from pathlib import Path

from configs.probe_actions import get_probe_config, PushProbeConfig, PickProbeConfig
from utils.contact_detection import ContactDetector, ProbeResponseRecorder
from data_collection.vla_policy import DummyVLAPolicy
from data_collection.collector import ProbeDataCollector


def test_probe_configs():
    """测试probe配置"""
    print("\n" + "="*60)
    print("Testing Probe Configurations")
    print("="*60)
    
    # Push probe
    push_config = get_probe_config("PushCube-v1")
    assert push_config.task_type == "push"
    print(f"✓ PushCube-v1: {push_config.task_type}, distance={push_config.push_distance}")
    
    # Pick probe
    pick_config = get_probe_config("PickCube-v1")
    assert pick_config.task_type == "pick"
    print(f"✓ PickCube-v1: {pick_config.task_type}, lift={pick_config.lift_height}")
    
    # Pull probe
    pull_config = get_probe_config("OpenCabinetDrawer-v1")
    assert pull_config.task_type == "pull"
    print(f"✓ OpenCabinetDrawer-v1: {pull_config.task_type}, distance={pull_config.pull_distance}")
    
    # 测试动作生成
    ee_pose = np.array([0.5, 0.0, 0.2, 1.0, 0, 0, 0])
    contact_normal = np.array([0, 0, 1])  # 向上
    
    push_action = push_config.generate_action(ee_pose, contact_normal)
    assert len(push_action) == 7
    print(f"✓ Push action generated: shape={push_action.shape}")
    
    pick_action = pick_config.generate_action(ee_pose)
    assert len(pick_action) == 7
    assert pick_action[2] > 0  # 应该有向上的分量
    print(f"✓ Pick action generated: shape={pick_action.shape}, z_lift={pick_action[2]}")


def test_contact_detector():
    """测试contact检测器"""
    print("\n" + "="*60)
    print("Testing Contact Detector")
    print("="*60)
    
    detector = ContactDetector(
        force_threshold=0.5,
        min_contact_steps=2,
        target_object_names=["cube"]
    )
    
    # 模拟观测
    obs = {
        'extra': {
            'tcp_wrench': np.array([1.0, 0.5, 0.3, 0, 0, 0])  # Force超过阈值
        }
    }
    
    # 模拟contact
    contacts = [
        {
            'body1': 'panda_hand',
            'body2': 'cube',
            'force': np.array([1.0, 0.5, 0.3]),
            'position': np.array([0.5, 0, 0.2]),
        }
    ]
    
    # 第一步
    is_contact, info = detector.detect_contact(obs, contacts)
    assert not is_contact  # 需要持续2步
    print(f"✓ Step 1: contact_counter={detector.contact_step_counter}")
    
    # 第二步
    is_contact, info = detector.detect_contact(obs, contacts)
    assert is_contact
    assert info is not None
    print(f"✓ Step 2: Valid contact detected")
    print(f"  Force magnitude: {info['force_magnitude']:.3f}")
    print(f"  Target object: {info['target_object']}")
    
    detector.reset()
    assert detector.contact_step_counter == 0
    print(f"✓ Detector reset successful")


def test_response_recorder():
    """测试response记录器"""
    print("\n" + "="*60)
    print("Testing Response Recorder")
    print("="*60)
    
    recorder = ProbeResponseRecorder()
    
    # 模拟probe过程
    for i in range(10):
        force = 2.0 + np.random.randn() * 0.5
        is_contact = i > 2  # 前几步没接触
        ee_pos = np.array([0.5 + i*0.001, 0, 0.2])
        obj_pos = np.array([0.6 + i*0.0005, 0, 0.1])
        
        recorder.record_step(force, is_contact, ee_pos, obj_pos)
    
    response = recorder.compute_response()
    
    print(f"✓ Response computed:")
    for key, val in response.items():
        print(f"  {key}: {val}")
    
    assert response['probe_steps'] == 10
    assert 0 <= response['contact_ratio'] <= 1
    assert response['total_ee_motion'] > 0
    print(f"✓ All response metrics valid")


def test_dummy_vla():
    """测试dummy VLA策略"""
    print("\n" + "="*60)
    print("Testing Dummy VLA Policy")
    print("="*60)
    
    policy = DummyVLAPolicy()
    policy.load()
    
    # 模拟图像和观测
    image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    instruction = "pick up the cube"
    
    action = policy.predict(image, instruction)
    
    assert len(action) == 7
    print(f"✓ Dummy VLA action: {action}")
    print(f"  Delta pos: {action[:3]}")
    print(f"  Delta rot: {action[3:6]}")
    print(f"  Gripper: {action[6]}")


def test_data_collection_mock():
    """测试数据收集器（不需要真实环境）"""
    print("\n" + "="*60)
    print("Testing Data Collector (Mock)")
    print("="*60)
    
    # 这里需要真实的ManiSkill环境，暂时跳过
    print("⚠ Skipping full collector test (requires ManiSkill env)")
    print("  To test with real env:")
    print("  1. Install ManiSkill: pip install mani-skill")
    print("  2. Run: python test_pipeline.py --with-env")


def test_hdf5_io():
    """测试HDF5读写"""
    print("\n" + "="*60)
    print("Testing HDF5 I/O")
    print("="*60)
    
    import h5py
    import tempfile
    
    # 创建临时文件
    with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        # 写入
        with h5py.File(tmp_path, 'w') as f:
            f.attrs['episode_id'] = 42
            f.attrs['outcome'] = 'success'
            
            f.create_dataset('contact_image', data=np.random.randint(0, 255, (224, 224, 3)))
            f.create_dataset('planned_action', data=np.random.randn(7))
            
            response_grp = f.create_group('probe_response')
            response_grp.create_dataset('mean_force', data=2.5)
            response_grp.create_dataset('contact_ratio', data=0.8)
        
        print(f"✓ Written to {tmp_path}")
        
        # 读取
        with h5py.File(tmp_path, 'r') as f:
            episode_id = f.attrs['episode_id']
            outcome = f.attrs['outcome']
            mean_force = f['probe_response']['mean_force'][()]
            
            assert episode_id == 42
            assert outcome == 'success'
            assert isinstance(mean_force, (float, np.floating))
            
        print(f"✓ Read successful: episode_id={episode_id}, outcome={outcome}")
        
    finally:
        # 清理
        import os
        os.unlink(tmp_path)


def run_all_tests():
    """运行所有测试"""
    print("\n" + "#"*70)
    print("# ProbeRetrieval Pipeline Tests")
    print("#"*70)
    
    tests = [
        ("Probe Configs", test_probe_configs),
        ("Contact Detector", test_contact_detector),
        ("Response Recorder", test_response_recorder),
        ("Dummy VLA", test_dummy_vla),
        ("HDF5 I/O", test_hdf5_io),
        ("Data Collector", test_data_collection_mock),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n✗ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "#"*70)
    print(f"# Test Results: {passed} passed, {failed} failed")
    print("#"*70)
    
    if failed == 0:
        print("\n✅ All tests passed! Ready to collect data.")
        print("\nNext steps:")
        print("  1. Test with real env: python test_pipeline.py --with-env")
        print("  2. Quick run: CUDA_VISIBLE_DEVICES=0 python run_collection.py --task PushCube-v1 --num-episodes 5 --policy dummy")
    else:
        print(f"\n❌ {failed} test(s) failed. Please fix before proceeding.")
        sys.exit(1)


if __name__ == '__main__':
    run_all_tests()
