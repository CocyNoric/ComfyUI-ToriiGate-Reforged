import { app } from "../../scripts/app.js";

const GROUNDING_NODE = "ToriiGate_GroundingBuilder_Reforged";
const ALWAYS_VISIBLE = new Set([
    "caption_type",
    "use_names",
    "add_tags",
    "add_character_list",
    "character_count",
    "add_character_tags",
    "add_character_descriptions",
]);

function isEnabled(value) {
    return value === true || value === 1 || value === "true";
}

function setElementVisible(element, visible) {
    if (!element) return;
    if (element.style) element.style.display = visible ? "" : "none";
    element.hidden = !visible;
}

function setWidgetVisible(widget, visible) {
    if (visible) {
        if (widget.__toriigateHidden) {
            widget.type = widget.__toriigateOriginalType;
            if (widget.__toriigateOriginalComputeSize === undefined) {
                delete widget.computeSize;
            } else {
                widget.computeSize = widget.__toriigateOriginalComputeSize;
            }
            delete widget.__toriigateOriginalType;
            delete widget.__toriigateOriginalComputeSize;
            delete widget.__toriigateHidden;
        }
        widget.hidden = false;
    } else {
        if (!widget.__toriigateHidden) {
            widget.__toriigateOriginalType = widget.type;
            widget.__toriigateOriginalComputeSize = widget.computeSize;
            widget.__toriigateHidden = true;
        }
        // converted-widget is ComfyUI's supported zero-height widget sentinel.
        // The former custom "hidden" type is still drawn by newer frontends.
        widget.type = "converted-widget";
        widget.computeSize = () => [0, -4];
        widget.hidden = true;
    }

    setElementVisible(widget.inputEl, visible);
    setElementVisible(widget.element, visible);
}

function widgetShouldBeVisible(widget, state) {
    if (ALWAYS_VISIBLE.has(widget.name)) return true;
    if (widget.name === "tags") return state.addTags;
    if (widget.name === "character_names") return state.addCharacterList;

    const match = /^char([1-5])_(name|tags|description)$/.exec(widget.name);
    if (!match) return true;

    const characterIndex = Number.parseInt(match[1], 10);
    if (characterIndex > state.characterCount) return false;
    if (match[2] === "tags") return state.addCharacterTags;
    if (match[2] === "description") return state.addCharacterDescriptions;
    return true;
}

function updateWidgets(node) {
    const widgets = node.widgets ?? [];
    if (widgets.length === 0) return false;

    const valueOf = (name) => widgets.find((widget) => widget.name === name)?.value;
    const parsedCount = Number.parseInt(valueOf("character_count"), 10);
    const state = {
        addTags: isEnabled(valueOf("add_tags")),
        addCharacterList: isEnabled(valueOf("add_character_list")),
        characterCount: Number.isFinite(parsedCount) ? Math.max(0, Math.min(5, parsedCount)) : 1,
        addCharacterTags: isEnabled(valueOf("add_character_tags")),
        addCharacterDescriptions: isEnabled(valueOf("add_character_descriptions")),
    };

    for (const widget of widgets) {
        setWidgetVisible(widget, widgetShouldBeVisible(widget, state));
    }

    const resize = () => {
        const computed = node.computeSize?.();
        if (computed && node.setSize) {
            const currentWidth = node.size?.[0] ?? computed[0];
            node.setSize([Math.max(currentWidth, computed[0]), Math.max(100, computed[1])]);
        }
        node.setDirtyCanvas?.(true, true);
        node.graph?.setDirtyCanvas?.(true, true);
    };
    if (typeof requestAnimationFrame === "function") requestAnimationFrame(resize);
    else setTimeout(resize, 0);
    return true;
}

function scheduleUpdate(node) {
    clearTimeout(node.__toriigateUpdateTimer);
    node.__toriigateUpdateTimer = setTimeout(() => updateWidgets(node), 0);
}

function installGroundingBehaviour(node, retryCount = 0) {
    if (node.__toriigateGroundingInstalled) {
        scheduleUpdate(node);
        return;
    }

    const widgets = node.widgets ?? [];
    if (widgets.length === 0) {
        if (retryCount < 10) {
            setTimeout(() => installGroundingBehaviour(node, retryCount + 1), 0);
        }
        return;
    }
    node.__toriigateGroundingInstalled = true;

    const toggleNames = new Set([
        "add_tags",
        "add_character_list",
        "character_count",
        "add_character_tags",
        "add_character_descriptions",
    ]);
    for (const widget of widgets) {
        if (!toggleNames.has(widget.name)) continue;
        const originalCallback = widget.callback;
        widget.callback = function () {
            const result = originalCallback?.apply(this, arguments);
            scheduleUpdate(node);
            return result;
        };
    }

    const originalOnAdded = node.onAdded;
    node.onAdded = function () {
        const result = originalOnAdded?.apply(this, arguments);
        scheduleUpdate(node);
        return result;
    };

    const originalOnConfigure = node.onConfigure;
    node.onConfigure = function () {
        const result = originalOnConfigure?.apply(this, arguments);
        scheduleUpdate(node);
        return result;
    };

    updateWidgets(node);
    setTimeout(() => updateWidgets(node), 50);
    setTimeout(() => updateWidgets(node), 200);
}

function isGroundingNode(node) {
    return node?.comfyClass === GROUNDING_NODE
        || node?.type === GROUNDING_NODE
        || node?.constructor?.comfyClass === GROUNDING_NODE;
}

app.registerExtension({
    // Keep this name distinct from the legacy ToriiGate extension so both
    // packages can be installed without one suppressing the other's hooks.
    name: "ComfyUI.ToriiGate.Reforged.GroundingBuilder",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== GROUNDING_NODE) return;
        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalOnNodeCreated?.apply(this, arguments);
            installGroundingBehaviour(this);
            return result;
        };
    },

    async nodeCreated(node) {
        if (isGroundingNode(node)) installGroundingBehaviour(node);
    },
});
