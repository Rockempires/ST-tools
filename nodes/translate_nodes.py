import os
import sys
import subprocess
import logging

# 配置日志 - 只显示基本信息
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 禁用所有 argostranslate 相关的日志
logging.getLogger('argostranslate').setLevel(logging.WARNING)
logging.getLogger('argostranslate.translate').setLevel(logging.WARNING)
logging.getLogger('argostranslate.sbd').setLevel(logging.WARNING)
logging.getLogger('argostranslate.utils').setLevel(logging.WARNING)
logging.getLogger('argostranslate.package').setLevel(logging.WARNING)

# 设置环境变量禁用 argostranslate 的调试模式
os.environ['ARGOS_DEBUG'] = '0'

# 导入后再禁用一次 argostranslate 的日志级别
try:
    import argostranslate
    import argostranslate.utils
    argostranslate.utils.logger.setLevel(logging.WARNING)
except ImportError:
    pass

# 全局变量
argostranslate_available = False
installed_version = None

def check_argostranslate_availability():
    """检查argostranslate是否可用并获取版本信息"""
    global argostranslate_available, installed_version
    try:
        import argostranslate
        argostranslate_available = True
        
        # 尝试获取版本信息
        try:
            installed_version = argostranslate.__version__
        except AttributeError:
            # 尝试从translate子模块获取版本
            try:
                from argostranslate import translate
                if hasattr(translate, '__version__'):
                    installed_version = translate.__version__
                else:
                    # 尝试通过pip show命令获取版本
                    try:
                        result = subprocess.run(
                            [sys.executable, '-m', 'pip', 'show', 'argostranslate'],
                            capture_output=True, text=True, timeout=10
                        )
                        for line in result.stdout.split('\n'):
                            if line.startswith('Version:'):
                                installed_version = line.split(':', 1)[1].strip()
                                break
                    except Exception as e:
                        logger.debug(f"通过pip获取版本时出错: {e}")
            except Exception as e:
                logger.debug(f"从translate模块获取版本时出错: {e}")
        
        return True
    except ImportError:
        argostranslate_available = False
        installed_version = None
        return False

def install_latest_argostranslate():
    """安装最新版本的argostranslate"""
    logger.info("正在安装argostranslate...")
    
    # 检测网络连接
    import socket
    try:
        socket.create_connection(("pypi.org", 443), timeout=10)
    except OSError:
        logger.error("网络无法连接服务器，需要科学上网")
        return False
    
    try:
        # 直接在当前进程中安装
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', 'argostranslate'],
            capture_output=True, text=True, timeout=60
        )
        
        if result.returncode == 0:
            logger.info("成功安装argostranslate")
            # 重新加载模块
            if 'argostranslate' in sys.modules:
                del sys.modules['argostranslate']
            # 尝试导入以验证安装
            try:
                import argostranslate
                # 尝试获取版本信息
                try:
                    version = argostranslate.__version__
                except AttributeError:
                    # 尝试从translate子模块获取版本
                    try:
                        from argostranslate import translate
                        if hasattr(translate, '__version__'):
                            version = translate.__version__
                        else:
                            version = "未知版本"
                    except Exception:
                        version = "未知版本"
                logger.info(f"成功安装argostranslate: {version}")
                global argostranslate_available, installed_version
                argostranslate_available = True
                installed_version = version
                return True
            except ImportError:
                logger.error("安装成功但无法导入argostranslate")
                return False
        else:
            logger.error(f"安装argostranslate失败: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"安装argostranslate时发生错误: {str(e)}")
        return False

def install_language_packs():
    """安装argostranslate语言包"""
    try:
        from argostranslate import package
        
        # 检查并安装中英文语言包
        languages = [('en', 'zh'), ('zh', 'en')]
        
        for from_code, to_code in languages:
            package_path = os.path.join(os.path.dirname(__file__), '..', 'models', f'translate-{from_code}_{to_code}-1_9.argosmodel')
            if os.path.exists(package_path):
                try:
                    # 使用正确的方法安装语言包
                    package.install_from_path(package_path)
                except Exception:
                    pass
    except Exception:
        pass

def install_dependencies():
    """安装依赖"""
    # 检查requirements.txt文件
    requirements_path = os.path.join(os.path.dirname(__file__), '..', 'requirements.txt')
    if not os.path.exists(requirements_path):
        logger.info("requirements.txt文件不存在，跳过依赖检测")
        return True
    
    with open(requirements_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    if not content:
        logger.info("requirements.txt文件为空，跳过依赖检测")
        return True
    
    # 检查argostranslate是否可用
    if not check_argostranslate_availability():
        # 尝试安装
        if not install_latest_argostranslate():
            return False
    
    # 安装语言包
    install_language_packs()
    
    return True

def check_dependency_at_startup():
    """启动时检查依赖"""
    check_argostranslate_availability()

# 启动时检查依赖
check_dependency_at_startup()

class ST_OfflineTranslator:
    """离线翻译节点 - 支持中文、英语和苗语之间的相互翻译"""
    DISPLAY_NAME = "离线翻译"
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        """定义节点输入参数"""
        languages = ["中文", "英语", "苗语"]
        return {
            "required": {
                "文本": ("STRING", {"default": "", "multiline": True}),
                "源语言": (languages, {"default": "中文"}),
                "目标语言": (languages, {"default": "英语"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "translate"
    CATEGORY = "🎯 石头工具"
    DESCRIPTION = "离线翻译节点，支持中文、英语和苗语之间的相互翻译"

    def translate(self, 文本, 源语言, 目标语言):
        """执行翻译操作"""
        try:
            # 延迟检查依赖，只有在首次使用时才执行
            if not argostranslate_available:
                logger.info("需要安装argostranslate依赖")
                if not install_dependencies():
                    return ("错误: 无法安装argostranslate依赖",)
            else:
                # 尝试导入argostranslate，确保依赖仍然存在
                try:
                    from argostranslate import translate
                except ImportError:
                    # 如果导入失败，重新尝试安装依赖
                    if not install_dependencies():
                        return ("错误: 无法安装argostranslate依赖",)
                    from argostranslate import translate
            
            # 检查语言包是否已安装
            installed_langs = translate.get_installed_languages()
            lang_codes = [lang.code for lang in installed_langs]
            if 'en' not in lang_codes or 'zh' not in lang_codes:
                install_language_packs()
                # 重新加载语言
                translate.load_installed_languages()
            
            # 语言代码映射
            lang_map = {
                "中文": "zh",
                "英语": "en",
                "苗语": "hmn"
            }
            
            from_code = lang_map.get(源语言, "zh")
            to_code = lang_map.get(目标语言, "en")
            
            # 执行翻译
            translation = translate.translate(文本, from_code, to_code)
            logger.info("翻译完成")
            return (translation,)
        except Exception as e:
            logger.error("翻译失败：没有字典模型")
            return ("翻译失败：没有字典模型",)