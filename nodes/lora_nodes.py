import json

from aiohttp import web
import folder_paths
import comfy.sd
import comfy.utils
from server import PromptServer

from .. import _reg_set, _reg_get


class _LoraInputs(dict):
    """仅接受本节点的动态字段；沿用 rgthree 的动态可选输入方式。"""

    def __contains__(self, key):
        return super().__contains__(key) or self._field(key) is not None

    @staticmethod
    def _field(key):
        for prefix in ("启用_", "LoRA_", "强度_"):
            if key.startswith(prefix) and key[len(prefix):].isdigit():
                return prefix
        if key.rstrip() in ("lora加载", "强度"):
            return "旧名称" if key.rstrip() == "lora加载" else "强度_"
        return None

    def __getitem__(self, key):
        if dict.__contains__(self, key):
            return super().__getitem__(key)
        field = self._field(key)
        if field == "启用_":
            return ("BOOLEAN", {"default": True})
        if field in ("LoRA_", "旧名称"):
            return (["无"] + folder_paths.get_filename_list("loras"), {"default": "无"})
        if field == "强度_":
            return ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01})
        raise KeyError(key)


class ST_LoraStack:
    """原生控件的动态多 LoRA 加载器。"""

    DISPLAY_NAME = "Lora加载"
    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("模型", "CLIP")
    FUNCTION = "load_loras"
    CATEGORY = "🎯 石头工具/基础流程"
    DESCRIPTION = "按顺序加载 LoRA；总开关关闭时直接返回输入，保留各路开关和参数。"

    @classmethod
    def INPUT_TYPES(cls):
        # 新参数为 optional，旧版接口工作流仍可直接执行。
        optional = _LoraInputs({
            "总开关": ("BOOLEAN", {"default": True, "label_on": "开启", "label_off": "关闭"}),
            "LoRA 数量": ("INT", {
                "default": 1, "min": 0, "step": 1,
                "tooltip": "设置显示的 LoRA 组数。可转换为输入；上游运行得到的数量在本节点执行时同步到界面，新增组默认不加载。",
            }),
            "CLIP": ("CLIP", {"tooltip": "可选的 CLIP 模型。"}),
        })
        for key in ("启用_1", "LoRA_1", "强度_1"):
            input_type, options = optional[key]
            optional[key] = (input_type, {**options, "socketless": True})
        return {
            "required": {"模型": ("MODEL",)},
            "optional": optional,
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    def load_loras(self, 模型, **kwargs):
        clip = kwargs.get("CLIP")
        count = max(0, int(kwargs.get("LoRA 数量", 4)))
        unique_id = kwargs.get("unique_id")
        if unique_id is not None:
            PromptServer.instance.send_sync("st-tools.lora-count", {"id": unique_id, "count": count})
        if not kwargs.get("总开关", True):
            return (模型, clip)

        old_model_info = _reg_get(模型)
        old_clip_info = _reg_get(clip) if clip is not None else {}
        lora_list = []
        # 缓存仅限本次执行，避免切换模型后节点长期持有所有权重。
        loaded_loras = {}
        for i in range(1, count + 1):
            suffix = " " * (i - 1)
            name = kwargs.get(f"LoRA_{i}", kwargs.get("lora加载" + suffix, "无"))
            strength = kwargs.get(f"强度_{i}", kwargs.get("强度" + suffix, 1.0))
            if not kwargs.get(f"启用_{i}", True) or not name or name == "无" or strength == 0:
                continue
            path = folder_paths.get_full_path_or_raise("loras", name)
            if path not in loaded_loras:
                loaded_loras[path] = comfy.utils.load_torch_file(path, safe_load=True)
            模型, clip = comfy.sd.load_lora_for_models(
                模型, clip, loaded_loras[path], strength, strength
            )
            lora_list.append((name, strength))

        # 模型克隆后迁移登记信息；本次加载记录不从旧对象累加。
        if lora_list:
            for key, value in old_model_info.items():
                if key != "lora_list":
                    _reg_set(模型, **{key: value})
            if clip is not None:
                for key, value in old_clip_info.items():
                    _reg_set(clip, **{key: value})
            _reg_set(模型, lora_list=lora_list)
        return (模型, clip)


# 浏览器必须参与动态控件布局。脚本由本文件直接提供，不生成额外 JS 文件。
_LORA_SCRIPT_PATH = "/st-tools/lora-nodes.js"
_LORA_SCRIPT_VERSION = "20261105-6"
_LORA_SCRIPT = r'''
import { app } from "../scripts/app.js";
import { api } from "../scripts/api.js";

const rowName = /^(启用|LoRA|强度)_\d+$/;
const normalizeCount = value => Math.max(0, Math.trunc(Number(value) || 0));

function rememberRows(node) {
    node._stLoraRows ??= {};
    for (const widget of node.widgets || []) {
        if (rowName.test(widget.name)) node._stLoraRows[widget.name] = widget.value;
    }
}

function removeRowWidget(node, widget) {
    // 使用节点的删除接口，清理新版前端中对应的控件注册状态。
    if (node.removeWidget) node.removeWidget(widget);
    else node.widgets.splice(node.widgets.indexOf(widget), 1);
}

function createStrengthRow(node, index, enabled) {
    const name = `强度_${index}`;
    let down = null;
    function setStrength(widget, value) {
        if (!Number.isFinite(value)) return;
        widget.value = Math.round(Math.max(-100, Math.min(100, value)) * 100) / 100;
        node.setDirtyCanvas(true, true);
        node.graph?.change();
    }
    // 下方一行使用原生画布控件样式，左侧强度、右侧开关，不创建网页界面。
    return node.addCustomWidget({
        name, type: "st_lora_strength", label: "强度",
        value: node._stLoraRows[name] ?? 1,
        options: {},
        computeSize: () => [0, LiteGraph.NODE_WIDGET_HEIGHT],
        serializeValue() { return this.value; },
        draw(ctx, owner, width, y, height) {
            this.last_y = y;
            this._rowWidth = width;
            const margin = 15;
            const split = Math.max(130, width - 72);
            const middle = y + height / 2;
            ctx.save();
            ctx.font = "14px Arial";
            ctx.textBaseline = "middle";
            ctx.lineWidth = 1;
            ctx.fillStyle = LiteGraph.WIDGET_BGCOLOR || "#222";
            ctx.strokeStyle = LiteGraph.WIDGET_OUTLINE_COLOR || "#666";
            ctx.beginPath();
            ctx.roundRect(margin, y, split - margin - 4, height, height / 2);
            ctx.fill();
            ctx.stroke();
            ctx.beginPath();
            ctx.roundRect(split + 4, y, width - split - margin - 4, height, height / 2);
            ctx.fill();
            ctx.stroke();
            ctx.fillStyle = LiteGraph.WIDGET_SECONDARY_TEXT_COLOR || "#999";
            ctx.textAlign = "left";
            ctx.fillText("强度", margin + 10, middle);
            ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR || "#ddd";
            ctx.textAlign = "right";
            ctx.fillText(Number(this.value).toFixed(2), split - 15, middle);
            ctx.textAlign = "left";
            ctx.fillStyle = enabled.value ? (LiteGraph.WIDGET_TEXT_COLOR || "#ddd") : (LiteGraph.WIDGET_SECONDARY_TEXT_COLOR || "#999");
            ctx.fillText(enabled.value ? "开" : "关", split + 8, middle);
            ctx.beginPath();
            ctx.fillStyle = enabled.value ? "#89a" : "#444";
            ctx.arc(width - margin - 10, middle, height * 0.3, 0, Math.PI * 2);
            ctx.fill();
            ctx.restore();
        },
        mouse(event, pos, owner) {
            const width = this._rowWidth || owner.size[0];
            const split = Math.max(130, width - 72);
            if (event.type === "pointerdown" || event.type === "mousedown") {
                if (pos[0] < 15 || pos[0] > width - 15) return false;
                if (pos[0] > split - 4 && pos[0] < split + 4) return false;
                down = {x: pos[0], moved: false, toggle: pos[0] >= split + 4};
                return true;
            }
            if (!down) return false;
            if (event.type === "pointermove" || event.type === "mousemove") {
                if (Math.abs(pos[0] - down.x) > 3) down.moved = true;
                return true;
            }
            if (event.type === "pointerup" || event.type === "mouseup") {
                const previous = down;
                down = null;
                if (previous.toggle) {
                    if (!previous.moved && pos[0] >= split + 4 && pos[0] <= width - 15) {
                        enabled.value = !enabled.value;
                        owner.setDirtyCanvas(true, true);
                        owner.graph?.change();
                    }
                } else if (!previous.moved && pos[0] >= 15 && pos[0] <= split - 4) {
                    app.canvas.prompt("强度", this.value, value => {
                        if (String(value).trim() !== "") setStrength(this, Number(value));
                    }, event);
                }
                return true;
            }
            return false;
        },
    });
}

function modelDefaultWidth(node) {
    // 通过模型加载节点的原生创建流程读取默认宽度，跟随前端的布局设置。
    const modelNode = LiteGraph.createNode("ST_ModelsLoader");
    return modelNode?.size[0] ?? node.computeSize()[0];
}

function resizeRows(node, value) {
    const count = normalizeCount(value);
    if (node._stLoraResizing || node._stLoraCount === count) return;
    const width = node._stLoraCount === undefined ? modelDefaultWidth(node) : node.size[0];
    node._stLoraResizing = true;
    try {
        rememberRows(node);
        // 保留仍在显示的控件对象，只增删发生变化的组，不替换整个控件数组。
        for (const widget of [...(node.widgets || [])].reverse()) {
            if (rowName.test(widget.name) && Number(widget.name.split("_")[1]) > count) {
                removeRowWidget(node, widget);
            }
        }
        for (let index = 1; index <= count; index++) {
            if (node.widgets.some(w => w.name === `LoRA_${index}`)) continue;
            const enabledName = `启用_${index}`;
            // 开关值独立保存和提交，显示及点击由下方的同一行控件负责。
            const enabled = node.addWidget("toggle", enabledName, node._stLoraRows[enabledName] ?? true, () => {}, {hidden: true});
            enabled.hidden = true;
            enabled.computeSize = () => [0, -4];
            const combo = node.addWidget("combo", `LoRA_${index}`, node._stLoraRows[`LoRA_${index}`] ?? "无", () => {}, {values: node._stLoraFiles});
            combo.label = "LoRA加载";
            createStrengthRow(node, index, enabled);
        }
        node._stLoraCount = count;
        node.setSize([width, node.computeSize()[1]]);
        node.setDirtyCanvas(true, true);
    } finally {
        node._stLoraResizing = false;
    }
}

function installNode(node) {
    if (node._stLoraInstalled) return;
    const counter = node.widgets?.find(w => w.name === "LoRA 数量");
    if (!counter) return;
    node._stLoraInstalled = true;
    const master = node.widgets.find(w => w.name === "总开关");
    if (master) Object.assign(master.options, {on: "开启", off: "关闭"});
    rememberRows(node);
    for (const widget of [...node.widgets].reverse()) {
        if (rowName.test(widget.name)) removeRowWidget(node, widget);
    }
    // 同时监听数值赋值和回调，覆盖箭头、手动输入及新版前端的数值更新路径。
    let owner = counter;
    let descriptor;
    while (owner && !descriptor) {
        descriptor = Object.getOwnPropertyDescriptor(owner, "value");
        owner = Object.getPrototypeOf(owner);
    }
    let current = counter.value;
    Object.defineProperty(counter, "value", {
        configurable: true, enumerable: true,
        get() { return descriptor?.get ? descriptor.get.call(this) : current; },
        set(value) {
            const count = normalizeCount(value);
            if (descriptor?.set) descriptor.set.call(this, count);
            else current = count;
            if (!node._stLoraConfiguring) resizeRows(node, count);
        },
    });
    const callback = counter.callback;
    counter.callback = function (value, ...args) {
        callback?.call(this, value, ...args);
        if (!node._stLoraConfiguring) resizeRows(node, value);
    };
    resizeRows(node, counter.value);
}

app.registerExtension({
    name: "ST_tools.DynamicLora",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "ST_LoraStack") return;
        nodeType.prototype._stLoraFiles = nodeData.input.optional.LoRA_1[0];
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = created?.apply(this, arguments);
            installNode(this);
            return result;
        };
        const changed = nodeType.prototype.onWidgetChanged;
        nodeType.prototype.onWidgetChanged = function (name, value) {
            const result = changed?.apply(this, arguments);
            if (name === "LoRA 数量" && !this._stLoraConfiguring) resizeRows(this, value);
            return result;
        };
        const configure = nodeType.prototype.configure;
        nodeType.prototype.configure = function (info) {
            installNode(this);
            this._stLoraConfiguring = true;
            try {
                let values = info.widgets_values;
                // 兼容最初的四组固定控件，及之前版本的三控件保存顺序。
                if (Array.isArray(values) && typeof values[0] === "string" && values.length === 8) {
                    values = [true, 4, ...values.flatMap((v, i) => i % 2 ? [v] : [true, v])];
                    info = {...info, widgets_values: values};
                }
                const saved = {...(info.properties?.st_lora_rows || {})};
                if (Array.isArray(values)) {
                    const count = normalizeCount(values[1] ?? 1);
                    for (let i = 1; i <= count; i++) {
                        const offset = 2 + (i - 1) * 3;
                        for (const [name, value] of [[`启用_${i}`, values[offset]], [`LoRA_${i}`, values[offset + 1]], [`强度_${i}`, values[offset + 2]]]) {
                            if (value !== undefined) saved[name] = value;
                        }
                    }
                    for (const widget of [...this.widgets].reverse()) {
                        if (rowName.test(widget.name)) removeRowWidget(this, widget);
                    }
                    this._stLoraRows = saved;
                    this._stLoraCount = undefined;
                    resizeRows(this, count);
                }
                // 移除旧版未连接的单组控件端口，避免旧序号残留在节点两侧。
                if (info.inputs) info = {...info, inputs: info.inputs.filter(input => !rowName.test(input.name) || input.link != null)};
                return configure.call(this, info);
            } finally {
                this._stLoraConfiguring = false;
                this.setSize([this.size[0], this.computeSize()[1]]);
                this.setDirtyCanvas(true, true);
            }
        };
        const serialize = nodeType.prototype.onSerialize;
        nodeType.prototype.onSerialize = function (info) {
            const result = serialize?.apply(this, arguments);
            rememberRows(this);
            info.properties ??= {};
            info.properties.st_lora_rows = {...this._stLoraRows};
            return result;
        };
    },
    nodeCreated(node) {
        // 构造完成后再初始化，兼容不调用原型创建钩子的前端版本。
        if (node.comfyClass === "ST_LoraStack") installNode(node);
    },
    setup() {
        api.addEventListener("st-tools.lora-count", event => {
            const node = app.graph?.getNodeById(event.detail.id);
            if (node?.comfyClass !== "ST_LoraStack") return;
            installNode(node);
            const counter = node.widgets.find(w => w.name === "LoRA 数量");
            if (counter) counter.value = event.detail.count;
            resizeRows(node, event.detail.count);
        });
    },
});
'''


@PromptServer.instance.routes.get(_LORA_SCRIPT_PATH)
async def _lora_script(request):
    return web.Response(
        text=_LORA_SCRIPT, content_type="application/javascript",
        headers={"Cache-Control": "no-store"},
    )


@web.middleware
async def _lora_extension(request, handler):
    response = await handler(request)
    # 新版前端通过带接口前缀的地址读取扩展列表，两个入口都要注入脚本。
    if request.path.rstrip("/") in ("/extensions", "/api/extensions") and response.status == 200:
        extensions = json.loads(response.body)
        script_url = f"{_LORA_SCRIPT_PATH}?v={_LORA_SCRIPT_VERSION}"
        if script_url not in extensions:
            extensions.append(script_url)
        return web.json_response(extensions)
    return response


PromptServer.instance.app.middlewares.append(_lora_extension)
