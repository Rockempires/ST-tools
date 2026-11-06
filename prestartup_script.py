import logging
import os
from pathlib import Path
import runpy

logging.basicConfig(level=logging.WARNING, format='[st-tools] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ComfyUI loads this entry point as a standalone script.
hotfix = Path(__file__).with_name("hotfix")
runpy.run_path(str(hotfix / "修复Windows连接重置.py"))["apply_asyncio_patch"]()

# ── 自动从 GitHub 单向同步 ──
def _start_auto_sync():
    """后台线程：ComfyUI 启动时自动拉取 origin/main 最新版。
    不阻塞启动；仅在工作区干净时快进到远端版本。
    """
    import threading
    def _worker():
        try:
            import subprocess
            repo = os.path.dirname(os.path.abspath(__file__))

            status = subprocess.run(
                ['git', 'status', '--porcelain'], cwd=repo, capture_output=True, timeout=10
            )
            if status.returncode != 0 or status.stdout:
                logger.debug("ST-tools 有本地改动，跳过自动同步")
                return

            # 1. fetch 远端引用
            r = subprocess.run(
                ['git', 'fetch', 'origin'],
                cwd=repo, capture_output=True, timeout=30
            )
            if r.returncode != 0:
                logger.debug(f"ST-tools auto-sync fetch failed: {r.stderr.decode(errors='ignore')[:200]}")
                return

            # 2. 比较本地 HEAD 与 origin/main
            local  = subprocess.run(['git', 'rev-parse', 'HEAD'],       cwd=repo, capture_output=True, text=True, timeout=10).stdout.strip()
            remote = subprocess.run(['git', 'rev-parse', 'origin/main'], cwd=repo, capture_output=True, text=True, timeout=10).stdout.strip()

            if local == remote:
                return  # 已是最新，跳过

            logger.info(f"[st-tools] 检测到新版本 {local[:7]} → {remote[:7]}，正在自动同步...")

            result = subprocess.run(
                ['git', 'merge', '--ff-only', 'origin/main'],
                cwd=repo, capture_output=True, timeout=30
            )
            if result.returncode == 0:
                logger.info("[st-tools] 自动同步完成 ✓")
            else:
                logger.debug(f"ST-tools 自动同步跳过: {result.stderr.decode(errors='ignore')[:200]}")


        except Exception as e:
            logger.debug(f"ST-tools auto-sync error: {e}")

    t = threading.Thread(target=_worker, daemon=True)
    t.start()

_start_auto_sync()
