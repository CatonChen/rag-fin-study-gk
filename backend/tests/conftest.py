import pytest
import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 设置测试环境变量
@pytest.fixture(autouse=True)
def setup_test_env():
    """设置测试环境变量"""
    os.environ["TESTING"] = "true"
    yield
    os.environ.pop("TESTING", None)

# 设置测试数据目录
@pytest.fixture
def test_data_dir():
    """返回测试数据目录路径"""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# 设置测试配置文件目录
@pytest.fixture
def test_config_dir():
    """返回测试配置文件目录路径"""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "config") 