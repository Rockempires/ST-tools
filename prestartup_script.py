"""
ST-tools 启动脚本
修复 Windows asyncio ConnectionResetError: [WinError 10054] 远程主机强迫关闭连接
"""

import os
import socket
import logging

logging.basicConfig(level=logging.WARNING, format='[st-tools] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def apply_asyncio_patch():
    """
    修复 _ProactorBasePipeTransport._call_connection_lost 在 socket.shutdown() 时
    未处理 ConnectionResetError 导致异常抛出的问题。
    """
    try:
        if os.name != 'nt':
            return True
        
        import asyncio
        from asyncio.proactor_events import _ProactorBasePipeTransport
        
        if hasattr(_ProactorBasePipeTransport._call_connection_lost, '_st_patched'):
            return True
        
        def _patched_call_connection_lost(self, exc):
            if getattr(self, '_called_connection_lost', False):
                return
            
            try:
                self._protocol.connection_lost(exc)
            except Exception:
                pass
            
            try:
                sock = getattr(self, '_sock', None)
                server = getattr(self, '_server', None)
                
                if sock is not None:
                    try:
                        if hasattr(sock, 'shutdown'):
                            try:
                                if sock.fileno() != -1:
                                    sock.shutdown(socket.SHUT_RDWR)
                            except (ConnectionResetError, OSError):
                                pass
                    except Exception:
                        pass
                    
                    try:
                        sock.close()
                    except Exception:
                        pass
                    
                    self._sock = None
                
                if server is not None:
                    try:
                        detach = getattr(server, '_detach', None)
                        if detach is not None:
                            try:
                                detach(self)
                            except TypeError:
                                try:
                                    detach()
                                except Exception:
                                    pass
                    except Exception:
                        pass
                    self._server = None
                    
            except Exception:
                pass
            finally:
                self._called_connection_lost = True
        
        _patched_call_connection_lost._st_patched = True
        _ProactorBasePipeTransport._call_connection_lost = _patched_call_connection_lost
        
        return True
        
    except Exception as e:
        logger.debug(f"asyncio patch error: {e}")
        return False

apply_asyncio_patch()

# ── 自动从 GitHub 单向同步 ──
def _start_auto_sync():
    """后台线程：ComfyUI 启动时自动拉取 origin/main 最新版。
    不阻塞启动；网络不可用时静默失败；本地未提交改动会被覆盖（单向同步=远端为准）。
    """
    import threading
    def _worker():
        try:
            import subprocess
            repo = os.path.dirname(os.path.abspath(__file__))

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

            # 3. 强制对齐（单向：远端为准）
            subprocess.run(['git', 'reset', '--hard', 'origin/main'], cwd=repo, capture_output=True, timeout=30)
            subprocess.run(['git', 'clean', '-fd'],                    cwd=repo, capture_output=True, timeout=30)

            logger.info("[st-tools] 自动同步完成 ✓")

        except Exception as e:
            logger.debug(f"ST-tools auto-sync error: {e}")

    t = threading.Thread(target=_worker, daemon=True)
    t.start()

_start_auto_sync()
