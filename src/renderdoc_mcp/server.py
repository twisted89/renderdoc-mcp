"""MCP server (mcp 2.x MCPServer) — registers all tools, resources, and prompts."""

from __future__ import annotations

import atexit
from importlib.metadata import PackageNotFoundError, version as _pkg_version

from mcp.server.mcpserver import MCPServer

from renderdoc_mcp.replay_thread import get_replay_thread, on_replay_thread
from renderdoc_mcp.session import get_session
from renderdoc_mcp.tools import (
    session_tools,
    event_tools,
    pipeline_tools,
    resource_tools,
    data_tools,
    shader_tools,
    advanced_tools,
    performance_tools,
    diagnostic_tools,
)

class RenderDocMCPServer(MCPServer):
    """MCPServer whose tools all execute on the single RenderDoc replay thread.

    mcp 2.x would otherwise run each sync tool on an arbitrary worker thread,
    concurrently with others, which RenderDoc's ReplayController does not allow.
    The decorator still returns the original function, so tools can keep
    calling each other directly.
    """

    def tool(self, *args, **kwargs):
        register_tool = super().tool(*args, **kwargs)

        def decorator(fn):
            register_tool(on_replay_thread(fn))
            return fn

        return decorator


def _server_version() -> str:
    try:
        return _pkg_version("renderdoc-mcp")
    except PackageNotFoundError:
        return "0.0.0"


mcp = RenderDocMCPServer(
    "renderdoc-mcp",
    instructions="RenderDoc graphics debugger MCP server — analyze GPU frame captures (.rdc) with AI",
    version=_server_version(),
)

# ── Register all tool modules ──

session_tools.register(mcp)
event_tools.register(mcp)
pipeline_tools.register(mcp)
resource_tools.register(mcp)
data_tools.register(mcp)
shader_tools.register(mcp)
advanced_tools.register(mcp)
performance_tools.register(mcp)
diagnostic_tools.register(mcp)


# ── Prompts ──

@mcp.prompt()
def debug_draw_call(event_id: int) -> str:
    """Deep-dive analysis of a single draw call.

    Guides through: pipeline state, shader bindings, constant buffers, render targets.
    """
    return f"""Analyze draw call at event {event_id} step by step:

1. First, call `set_event` with event_id={event_id} and `get_action` to understand what this draw call does.
2. Call `get_pipeline_state` to inspect the full pipeline configuration:
   - Check topology, viewports, scissors
   - Check rasterizer state (cull mode, fill mode)
   - Check depth/stencil state
   - Check blend state
3. Call `get_vertex_inputs` to see vertex layout and buffer bindings.
4. For each bound shader stage (vertex, pixel, etc.):
   a. Call `get_shader_bindings` to see what resources are bound
   b. Call `get_shader_reflection` to understand the shader interface
   c. For each constant buffer, call `get_cbuffer_contents` to read actual values
5. Check the render target outputs — if textures are bound, consider using `save_texture`
   or `pick_pixel` to inspect the results.
6. Summarize findings: what this draw call renders, any potential issues found."""


@mcp.prompt()
def find_rendering_issue(description: str) -> str:
    """Guide AI through diagnosing a rendering problem.

    Start from a user description and systematically check common causes.
    """
    return f"""The user reports a rendering issue: "{description}"

Investigate step by step:

1. First call `get_capture_info` to check GPU/driver and known_gpu_quirks.
2. Call `list_actions` (use filter= or event_type= to narrow down) to find relevant passes.
3. For suspected draw calls use `get_draw_call_state` for a one-shot overview.
4. Deepen with `get_pipeline_state` — check depth, blend, cull, viewports.
5. Check shader bindings with `get_shader_bindings`; read constants with `get_cbuffer_contents`
   (use filter= to focus on specific variables like "ibl", "exposure", "taa").
6. Use `disassemble_shader` with search= to find relevant code (e.g. search="SampleSH").
7. For color/lighting anomalies:
   - `sample_pixel_region` to scan the RT for NaN/Inf/negative values
   - `pixel_history` on suspicious pixels to trace all writes
   - `debug_shader_at_pixel` on an anomalous pixel for step-by-step trace
8. For reflection issues: `diagnose_reflection_mismatch` and `diff_draw_calls`
9. For negative-value / flash issues: `diagnose_negative_values`
10. Summarize root cause with fix suggestions."""


@mcp.prompt()
def analyze_performance() -> str:
    """Performance analysis workflow for a captured frame."""
    return """Analyze the frame for performance issues:

1. Call `get_capture_info` to check GPU/driver and known quirks.
2. Call `get_frame_overview` for draw count, clear count, RT list.
3. Call `get_pass_timing` (granularity="pass") to find the most expensive passes.
4. Call `analyze_overdraw` to check fill-rate pressure.
5. Call `analyze_bandwidth` to estimate memory bandwidth.
6. Call `analyze_state_changes` to find draw call batching opportunities.
7. Call `diagnose_mobile_risks` with check_categories=["performance","precision","compatibility"]
   for a prioritized risk list.
8. For the most expensive pass:
   a. `get_pipeline_state` — unnecessary state?
   b. `get_vertex_inputs` — vertex format optimality?
   c. `list_textures` — oversized textures?
9. Summarize with actionable optimizations ranked by impact."""



# ── Cleanup ──

def _cleanup():
    # RenderDoc must be shut down on the same thread that used it.
    replay = get_replay_thread()
    try:
        if replay.is_alive:
            replay.submit(get_session().shutdown).result(timeout=30)
    except Exception:
        pass

atexit.register(_cleanup)


# ── Entry point ──

def main():
    mcp.run()


if __name__ == "__main__":
    main()
