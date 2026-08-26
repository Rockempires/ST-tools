"""
ST-tools 启动脚本
修复 Windows asyncio ConnectionResetError: [WinError 10054] 远程主机强迫关闭连接
"""

import os
import socket
import logging

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
