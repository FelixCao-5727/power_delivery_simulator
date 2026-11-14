from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import plotly.graph_objects as go

from simulator_core_v23 import DSRDashboardBackend, Position


NODE_TEXT_POSITIONS = {
    1: "top left",
    2: "top center",
    3: "top right",
    4: "middle left",
    5: "middle center",
    6: "middle right",
    7: "bottom left",
    8: "bottom center",
    9: "bottom right",
}


def _add_polyline_trace(
    figure: go.Figure,
    positions: Iterable[Position],
    *,
    color: str,
    width: float,
    dash: str = "solid",
    hover_text: str = "",
    name: str = "",
    legendgroup: str | None = None,
    showlegend: bool = False,
    opacity: float = 1.0,
) -> None:
    points = list(positions)
    if not points:
        return

    figure.add_trace(
        go.Scatter(
            x=[point[0] for point in points],
            y=[point[1] for point in points],
            mode="lines",
            line={"color": color, "width": width, "dash": dash},
            hovertemplate=(hover_text + "<extra></extra>") if hover_text else None,
            hoverinfo="text" if hover_text else "skip",
            name=name,
            legendgroup=legendgroup,
            showlegend=showlegend,
            opacity=opacity,
        )
    )


def build_grid_figure(backend: DSRDashboardBackend) -> go.Figure:
    """Build a layered Plotly figure directly from simulator state."""

    if backend.graph is None:
        figure = go.Figure()
        figure.update_layout(
            annotations=[{
                "text": "Map data are not loaded.",
                "x": 0.5,
                "y": 0.5,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 18},
            }],
        )
        return figure

    graph = backend.graph
    figure = go.Figure()

    # Layer 1: one matrix-backed unknown-Pixel background.
    x_coordinates = list(range(graph.x_min, graph.x_max + 1))
    y_coordinates = list(range(graph.y_min, graph.y_max + 1))
    unknown_matrix = [
        [
            0 if graph.get_pixel((x, y)).is_inspected(backend.event_id) else 1
            for x in x_coordinates
        ]
        for y in y_coordinates
    ]
    pixel_knowledge = [
        [
            "Known" if graph.get_pixel((x, y)).is_inspected(backend.event_id) else "Unknown"
            for x in x_coordinates
        ]
        for y in y_coordinates
    ]
    figure.add_trace(
        go.Heatmap(
            x=x_coordinates,
            y=y_coordinates,
            z=unknown_matrix,
            customdata=pixel_knowledge,
            zmin=0,
            zmax=1,
            colorscale=[
                [0.0, "rgba(255,255,255,0.0)"],
                [0.499, "rgba(255,255,255,0.0)"],
                [0.5, "rgba(242,243,245,0.88)"],
                [1.0, "rgba(242,243,245,0.88)"],
            ],
            showscale=False,
            hovertemplate=(
                "Pixel: (%{x}, %{y})<br>Knowledge: %{customdata}"
                "<extra></extra>"
            ),
            xgap=1,
            ygap=1,
            name="Pixels",
        )
    )

    # Subtle square grid at Pixel boundaries.
    shapes = []
    for x in np.arange(graph.x_min - 0.5, graph.x_max + 1.0, 1.0):
        shapes.append({
            "type": "line",
            "x0": x,
            "x1": x,
            "y0": graph.y_min - 0.5,
            "y1": graph.y_max + 0.5,
            "line": {"color": "rgba(160,165,170,0.28)", "width": 0.6, "dash": "dot"},
            "layer": "below",
        })
    for y in np.arange(graph.y_min - 0.5, graph.y_max + 1.0, 1.0):
        shapes.append({
            "type": "line",
            "x0": graph.x_min - 0.5,
            "x1": graph.x_max + 0.5,
            "y0": y,
            "y1": y,
            "line": {"color": "rgba(160,165,170,0.28)", "width": 0.6, "dash": "dot"},
            "layer": "below",
        })

    # Layer 2: roads. Electrical-network Pixels are implicit Level 1 roads.
    for line in backend.electrical_lines.values():
        _add_polyline_trace(
            figure,
            line.position,
            color="#9AA0A6",
            width=7.0,
            hover_text=f"Implicit Level 1 road under {line.name}",
            name="Level 1 Road",
            legendgroup="road-level-1",
        )

    for road in sorted(backend.roads.values(), key=lambda item: item.level):
        if road.level == 1:
            _add_polyline_trace(
                figure,
                road.position,
                color="#9AA0A6",
                width=7.0,
                hover_text=f"{road.name}<br>Road Level 1",
                name="Level 1 Road",
                legendgroup="road-level-1",
            )
        else:
            _add_polyline_trace(
                figure,
                road.position,
                color="#858B91",
                width=11.0,
                hover_text=f"{road.name}<br>Road Level 2",
                name="Level 2 Road",
                legendgroup="road-level-2",
            )
            _add_polyline_trace(
                figure,
                road.position,
                color="#C6CACE",
                width=1.8,
                dash="dash",
                hover_text=f"{road.name}<br>Road Level 2",
                name="Level 2 Centerline",
                legendgroup="road-level-2",
            )

    # Static unknown-region boundaries remain visible while Pixel fills disappear.
    if backend.unknown_state is not None:
        for region in backend.unknown_state.unknown_regions:
            x_min, y_min = region.lower_left
            x_max, y_max = region.upper_right
            shapes.append({
                "type": "rect",
                "x0": x_min - 0.5,
                "x1": x_max + 0.5,
                "y0": y_min - 0.5,
                "y1": y_max + 0.5,
                "fillcolor": "rgba(0,0,0,0)",
                "line": {"color": "rgba(150,155,160,0.75)", "width": 1.0},
                "layer": "below",
            })

    # Layer 3: electrical lines and labels.
    annotations = []
    for line in backend.electrical_lines.values():
        knowledge = line.knowledge_state(backend.event_id)
        hover_text = (
            f"{line.name}<br>Knowledge: {knowledge}"
            f"<br>State: {line.state}<br>Power: "
            f"{'Energized' if line.energized else 'De-energized'}"
            f"<br>Area: {line.area_name or '-'}"
        )
        if knowledge == "known":
            _add_polyline_trace(
                figure,
                line.position,
                color="#18C937" if line.energized else "#777C82",
                width=3.2,
                dash="solid",
                hover_text=hover_text,
                name=line.name,
            )
        else:
            for first_pixel, second_pixel in zip(
                line.physical_pixels[:-1], line.physical_pixels[1:]
            ):
                _add_polyline_trace(
                    figure,
                    [first_pixel.position, second_pixel.position],
                    color="#F0A000",
                    width=3.0,
                    dash="dash",
                    hover_text=hover_text,
                    name=line.name,
                )

        label_x, label_y, _, _ = line._get_label_location(label_gap=0.30)
        annotations.append({
            "x": label_x,
            "y": label_y,
            "text": line.name,
            "showarrow": False,
            "font": {"color": "#20252B", "size": 13, "family": "Consolas, monospace"},
        })

    # Active route layer for units that are already Moving.
    for unit in backend.mobile_units.values():
        if unit.state == "Moving" and unit.planned_path:
            active_path = [unit.position, *unit.planned_path]
            _add_polyline_trace(
                figure,
                active_path,
                color="#2678FF",
                width=2.5,
                dash="dash",
                hover_text=f"{unit.name} active route",
                name=f"{unit.name} route",
            )
            previous_x, previous_y = active_path[-2]
            end_x, end_y = active_path[-1]
            annotations.append({
                "x": end_x, "y": end_y,
                "ax": previous_x, "ay": previous_y,
                "xref": "x", "yref": "y", "axref": "x", "ayref": "y",
                "text": "", "showarrow": True, "arrowhead": 3,
                "arrowsize": 1.2, "arrowwidth": 2.5,
                "arrowcolor": "#2678FF",
            })

    # Layer 4: nodes.
    source = getattr(backend, "transmission_source", None)
    source_node_name = source.node_name if source is not None else None
    node_degree = {node_name: 0 for node_name in backend.nodes}
    for line in backend.electrical_lines.values():
        for endpoint in line.end_nodes:
            if endpoint in node_degree:
                node_degree[endpoint] += 1
    for node in backend.nodes.values():
        # The grid-source node gets the IEC AC-source symbol below.
        if node.name == source_node_name:
            continue
        knowledge = node.knowledge_state(backend.event_id)
        marker_color = (
            "#F0A000" if knowledge != "known"
            else ("#18C937" if node.energized else "#777C82")
        )
        load_text = "No load"
        if node.has_load:
            load_text = (
                f"Load: P={node.active_power:g}, Q={node.reactive_power:g}, "
                f"weight={node.load_weight:g}"
            )
        figure.add_trace(
            go.Scatter(
                x=[node.position[0]],
                y=[node.position[1]],
                mode="markers+text",
                marker={
                    "symbol": "circle" if node_degree[node.name] > 1 else "circle-open",
                    "size": 11 if node_degree[node.name] > 1 else 14,
                    "color": marker_color if node_degree[node.name] > 1 else "rgba(255,255,255,0)",
                    "line": {"color": marker_color, "width": 2.5},
                },
                text=[node.name],
                textposition=NODE_TEXT_POSITIONS.get(node.label_position, "top center"),
                textfont={"color": "#20252B", "size": 14},
                hovertemplate=(
                    f"{node.name}<br>Position: {node.position}"
                    f"<br>Knowledge: {knowledge}"
                    f"<br>Power: {'Energized' if node.energized else 'De-energized'}"
                    f"<br>{load_text}<br>Area: {node.area_name or 'Transmission'}"
                    "<extra></extra>"
                ),
                showlegend=False,
                name=node.name,
            )
        )

    # Switches use IEC-style open/closed contact blades over the conductor.
    for switch in backend.switches.values():
        center_x, center_y = switch.get_center_position(backend.nodes, distance_from_node=1.6)
        line = switch._line
        node_x, node_y = backend.nodes[switch.node_name].position
        adjacent_x, adjacent_y = (
            line.position[1] if switch.node_name == line.node_a else line.position[-2]
        )
        direction_norm = float(np.hypot(adjacent_x - node_x, adjacent_y - node_y)) or 1.0
        ux = (adjacent_x - node_x) / direction_norm
        uy = (adjacent_y - node_y) / direction_norm
        px, py = -uy, ux
        half_span = 0.80
        first = (center_x - ux * half_span, center_y - uy * half_span)
        second = (center_x + ux * half_span, center_y + uy * half_span)
        switch_color = {
            "closed": "#18C937",
            "opened": "#E02020",
            "unknown": "#F0A000",
        }[switch.state]
        blade_end = second if switch.state == "closed" else (
            center_x + ux * 0.48 + px * 0.48,
            center_y + uy * 0.48 + py * 0.48,
        )
        figure.add_trace(go.Scatter(
            x=[first[0], second[0]], y=[first[1], second[1]],
            mode="markers", marker={
                "symbol": "circle-open", "size": 11,
                "color": "#111111", "line": {"color": "#111111", "width": 2.7},
            }, hoverinfo="skip", showlegend=False, name=f"{switch.name} contacts",
        ))
        figure.add_trace(go.Scatter(
            x=[first[0], blade_end[0]], y=[first[1], blade_end[1]],
            mode="lines", line={"color": switch_color, "width": 3},
            hoverinfo="skip", showlegend=False, name=f"{switch.name} blade",
        ))
        figure.add_trace(
            go.Scatter(
                x=[center_x],
                y=[center_y],
                mode="markers+text",
                marker={"size": 48, "opacity": 0},
                text=[switch.name],
                textposition=NODE_TEXT_POSITIONS.get(switch.label_position, "top center"),
                textfont={"color": "#111111", "size": 13},
                hovertemplate=(
                    f"{switch.name}<br>Current state: {switch.state}"
                    f"<br>Node: {switch.node_name}<br>Line: {switch.line_name}"
                    "<extra></extra>"
                ),
                showlegend=False,
                name=switch.name,
            )
        )

    # Layer 5: known active faults.
    for fault in backend.faults.values():
        if not fault.active or not fault.is_known(backend.event_id):
            continue
        public_required = fault.public_repair_resource
        public_remaining = fault.public_remaining_repair_resource
        analysis_status = "Complete" if fault.analyzed else "Pending"
        required_text = (
            str(public_required) if public_required is not None else "Hidden"
        )
        remaining_text = (
            str(public_remaining) if public_remaining is not None else "Hidden"
        )
        fault_hover = (
            f"Fault {fault.name}<br>Position: {fault.position}"
            f"<br>Status: Active"
            f"<br>Analysis: {analysis_status}"
            f"<br>Required repair resource: {required_text}"
            f"<br>Remaining repair resource: {remaining_text}"
            f"<br>Area: {', '.join(fault.area_names)}"
            "<extra></extra>"
        )

        figure.add_trace(
            go.Scatter(
                x=[fault.position[0]],
                y=[fault.position[1]],
                mode="markers+text",
                marker={"size": 24, "opacity": 0},
                text=[fault.name],
                textposition="top right",
                textfont={"color": "#7F0000", "size": 14},
                hovertemplate=fault_hover,
                showlegend=False,
                name=f"Fault {fault.name}",
            )
        )
        fault_x, fault_y = fault.position
        shapes.append({
            "type": "path",
            "path": (
                f"M {fault_x - 0.80},{fault_y - 0.80} "
                f"L {fault_x + 0.80},{fault_y + 0.80} "
                f"M {fault_x - 0.80},{fault_y + 0.80} "
                f"L {fault_x + 0.80},{fault_y - 0.80}"
            ),
            "line": {"color": "#7F0000", "width": 3},
            "layer": "above",
        })

    # IEC-style AC source: circle with a sine-wave mark at the grid node.
    if source is not None:
        source_node = backend.nodes[source.node_name]
        source_color = "#E02020" if source.fault_active else "#18C937"
        figure.add_trace(
            go.Scatter(
                x=[source_node.position[0]],
                y=[source_node.position[1]],
                mode="markers+text",
                marker={
                    "symbol": "circle",
                    "size": 18,
                    "color": "rgba(255,255,255,0)",
                    "line": {"color": source_color, "width": 2},
                },
                text=[source_node.name],
                textposition=NODE_TEXT_POSITIONS.get(source_node.label_position, "top center"),
                textfont={"color": "#20252B", "size": 14},
                hovertemplate=(
                    f"{source.name}<br>Status: {'Outage' if source.fault_active else 'Available'}"
                    f"<br>Node: {source_node.name}"
                    f"<br>Restoration time step: {source.restoration_time_step}"
                    f"<br>Grid P/Q: {source.active_power:g}/{source.reactive_power:g}"
                    f"<br>Isolated switches: {', '.join(source.isolated_switch_names) or '-'}"
                    "<extra></extra>"
                ),
                showlegend=False,
                name=source.name,
            )
        )
        annotations.append({
            "x": source_node.position[0], "y": source_node.position[1],
            "text": "~", "showarrow": False,
            "font": {"color": source_color, "size": 15, "family": "Consolas, monospace"},
        })
        if source.fault_active:
            annotations.append({
                "x": source_node.position[0] + 0.35,
                "y": source_node.position[1] + 0.35,
                "text": "OUTAGE", "showarrow": False,
                "bgcolor": "rgba(255,255,255,0)", "bordercolor": "rgba(255,255,255,0)",
                "borderpad": 0,
                "font": {"color": "#FFFFFF", "size": 12},
            })

    # Offset crowded mobile badges, with a leader to each unit's true location.
    switch_obstacles = [
        switch.get_center_position(backend.nodes, distance_from_node=1.6)
        for switch in backend.switches.values()
    ]
    fixed_obstacles = [node.position for node in backend.nodes.values()] + switch_obstacles
    placed_mobile_icons = []
    unit_colors = {"RC": "#4AA3FF", "PAC": "#B56CE3", "MPS": "#F2CC16"}
    for unit in backend.mobile_units.values():
        x, y = unit.position
        display_x, display_y = x, y
        obstacles = [*fixed_obstacles, *placed_mobile_icons]
        nearest_clearance = min(
            (float(np.hypot(x - ox, y - oy)) for ox, oy in obstacles),
            default=float("inf"),
        )
        if nearest_clearance < 2.4:
            candidates = [
                (x + dx, y + dy)
                for dx, dy in (
                    (0, 2.8), (1.98, 1.98), (2.8, 0), (1.98, -1.98),
                    (0, -2.8), (-1.98, -1.98), (-2.8, 0), (-1.98, 1.98),
                )
            ]
            display_x, display_y = max(
                candidates,
                key=lambda candidate: min(
                    float(np.hypot(candidate[0] - ox, candidate[1] - oy))
                    for ox, oy in obstacles
                ),
            )
            direction_x = x - display_x
            direction_y = y - display_y
            direction_norm = float(np.hypot(direction_x, direction_y)) or 1.0
            direction_x /= direction_norm
            direction_y /= direction_norm
            edge_distance = min(
                1.16 / abs(direction_x) if direction_x else float("inf"),
                0.72 / abs(direction_y) if direction_y else float("inf"),
            )
            arrow_start = (
                display_x + direction_x * (edge_distance + 0.05),
                display_y + direction_y * (edge_distance + 0.05),
            )
            figure.add_trace(go.Scatter(
                x=[arrow_start[0], x], y=[arrow_start[1], y], mode="lines",
                line={"color": unit_colors.get(unit.unit_type, "#20252B"), "width": 3.2},
                hoverinfo="skip", showlegend=False, name=f"{unit.name} leader",
            ))
            figure.add_trace(go.Scatter(
                x=[x], y=[y], mode="markers",
                marker={
                    "symbol": "triangle-up", "size": 14,
                    "angle": float(np.degrees(np.arctan2(direction_x, direction_y))) % 360,
                    "color": unit_colors.get(unit.unit_type, "#20252B"),
                    "line": {"color": unit_colors.get(unit.unit_type, "#20252B"), "width": 1},
                }, hoverinfo="skip", showlegend=False, name=f"{unit.name} leader arrow",
            ))
        placed_mobile_icons.append((display_x, display_y))
        r = unit.inspection_range
        shapes.append({
            "type": "circle",
            "x0": x - r - 0.5,
            "x1": x + r + 0.5,
            "y0": y - r - 0.5,
            "y1": y + r + 0.5,
            "fillcolor": "rgba(0,0,0,0)",
            "line": {"color": unit.range_color, "width": 1.8, "dash": "dash"},
            "layer": "above",
        })
        is_mps = unit.unit_type == "MPS"
        if unit.unit_type == "RC":
            resource_text = f"{unit.remaining_resources}/{unit.total_resources}"
            extra_text = f"<br>Repair resources: {resource_text}"
        elif unit.unit_type == "MPS":
            resource_text = f"{unit.energy:g}/{unit.energy_capacity:g} kW.step"
            extra_text = (
                f"<br>Energy: {resource_text}"
                f"<br>Output: P={unit.current_p:g} kW, Q={unit.current_q:g} kVar"
                f"<br>Limits: P={unit.p_limit:g} kW, S={unit.s_limit:g} kVA"
                f"<br>Connected Areas: {', '.join(unit.connected_area_names) or '-'}"
            )
        else:
            resource_text = "N/A"
            extra_text = ""
        figure.add_trace(
            go.Scatter(
                x=[display_x],
                y=[display_y],
                mode="markers+text",
                marker={"size": 48, "opacity": 0},
                text=[""],
                hovertemplate=(
                    f"{unit.name}<br>Type: {unit.unit_type}"
                    f"<br>State: {unit.state}<br>Position: {unit.position}"
                    f"<br>Move speed: {unit.move_speed}"
                    f"<br>Inspection range: {unit.inspection_range}"
                    f"{extra_text}<extra></extra>"
                ),
                showlegend=False,
                name=unit.name,
            )
        )
        half_width, half_height, corner = 1.16, 0.72, 0.24
        shapes.append({
            "type": "path",
            "path": (
                f"M {display_x - half_width + corner},{display_y - half_height} "
                f"L {display_x + half_width - corner},{display_y - half_height} "
                f"Q {display_x + half_width},{display_y - half_height} {display_x + half_width},{display_y - half_height + corner} "
                f"L {display_x + half_width},{display_y + half_height - corner} "
                f"Q {display_x + half_width},{display_y + half_height} {display_x + half_width - corner},{display_y + half_height} "
                f"L {display_x - half_width + corner},{display_y + half_height} "
                f"Q {display_x - half_width},{display_y + half_height} {display_x - half_width},{display_y + half_height - corner} "
                f"L {display_x - half_width},{display_y - half_height + corner} "
                f"Q {display_x - half_width},{display_y - half_height} {display_x - half_width + corner},{display_y - half_height} Z"
            ),
            "fillcolor": unit_colors.get(unit.unit_type, "#FFFFFF"),
            "line": {"color": "#20252B", "width": 2.5},
            "layer": "above",
        })
        annotations.append({
            "x": display_x, "y": display_y + half_height + 0.1, "text": unit.name,
            "showarrow": False, "yanchor": "bottom",
            "font": {"color": "#20252B", "size": 13},
        })
        annotations.append({
            "x": display_x, "y": display_y, "text": f"<b>{unit.unit_type}</b>",
            "showarrow": False,
            "font": {"color": "#20252B", "size": 14},
        })

    figure.update_layout(
        font={"family": "Consolas, monospace", "size": 14},
        hoverlabel={"font": {"family": "Consolas, monospace"}},
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        margin={"l": 36, "r": 12, "t": 8, "b": 34},
        hovermode="closest",
        showlegend=False,
        shapes=shapes,
        annotations=annotations,
        uirevision="dsr-grid-v23",
        dragmode="pan",
        xaxis={
            "range": [graph.x_min - 0.5, graph.x_max + 0.5],
            "tickmode": "linear",
            "tick0": graph.x_min + (graph.x_min % 2),
            "dtick": 2,
            "title": "X",
            "showgrid": False,
            "zeroline": False,
            "constrain": "domain",
            "constraintoward": "center",
        },
        yaxis={
            "range": [graph.y_min - 0.5, graph.y_max + 0.5],
            "tickmode": "linear",
            "tick0": graph.y_min + (graph.y_min % 2),
            "dtick": 2,
            "title": "Y",
            "showgrid": False,
            "zeroline": False,
            "constrain": "domain",
            "constraintoward": "middle",
            "scaleanchor": "x",
            "scaleratio": 1,
        },
    )
    return figure
