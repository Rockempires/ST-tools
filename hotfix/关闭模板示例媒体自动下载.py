"""在 ComfyUI 选定前端后识别模板加载逻辑，让官方模板跳过示例媒体下载并直接打开工作流。"""

import logging
from pathlib import Path
import re

from aiohttp import web
from server import PromptServer


logger = logging.getLogger(__name__)

PREPARING_MEDIA = b"templateWorkflows.preparingMedia"
IDENTIFIER = rb"[$\w]+"
CURRENT = re.compile(
    rb"json:(?P<fetched>" + IDENTIFIER + rb")\.strict\?"
    rb"await withPreparedSampleInputs\((?P=fetched)\.json,"
    + IDENTIFIER + rb"," + IDENTIFIER + rb"\):(?P<json>" + IDENTIFIER
    + rb"),template:"
)
LEGACY = re.compile(
    rb'if\(' + IDENTIFIER + rb'!=="default"\|\|!(?P<template>'
    + IDENTIFIER + rb')\?\.io\?\.inputs\?\.length\)'
    rb'(?P<result>return\{json:' + IDENTIFIER + rb',template:(?P=template)\});'
)
LOADER = re.compile(
    rb"async function loadTemplateData\([^)]*\)\{.*?"
    rb"(?=async function loadTemplateGraph\()",
    re.DOTALL,
)


def skip_sample_preparation(source):
    loaders = list(LOADER.finditer(source))
    if len(loaders) != 1:
        return None

    loader = loaders[0]
    body = loader.group()
    current = list(CURRENT.finditer(body))
    legacy = list(LEGACY.finditer(body))
    if len(current) == 1 and not legacy:
        replacement = CURRENT.sub(rb"json:\g<json>,template:", body)
    elif len(legacy) == 1 and not current and PREPARING_MEDIA in body:
        guard = legacy[0]
        if b"await prepareTemplateInputs(" not in body[guard.end():]:
            return None
        replacement = body[:guard.start()] + guard["result"] + b"}"
    else:
        return None

    return source[:loader.start()] + replacement + source[loader.end():]


def install(server):
    matches = []
    for bundle in (Path(server.web_root) / "assets").glob("*.js"):
        source = bundle.read_bytes()
        if PREPARING_MEDIA in source and b"async function loadTemplateData(" in source:
            matches.append((bundle, source))

    if len(matches) == 1:
        bundle, source = matches[0]
        patched = skip_sample_preparation(source)
        if patched is not None:
            async def serve_bundle(request):
                return web.Response(
                    body=patched,
                    content_type="text/javascript",
                    headers={"Cache-Control": "no-store"},
                )

            server.app.router.add_get(f"/assets/{bundle.name}", serve_bundle)
            logger.info("[模板媒体补丁] 已关闭示例媒体自动下载（%s）", bundle.name)
            return

    logger.warning("[模板媒体补丁] 无法识别当前前端的模板加载逻辑，未应用补丁")


def apply_template_media_patch():
    if PromptServer.instance is None:
        return
    try:
        install(PromptServer.instance)
    except (OSError, RuntimeError, ValueError) as error:
        logger.warning("[模板媒体补丁] 无法应用补丁：%s", error)
