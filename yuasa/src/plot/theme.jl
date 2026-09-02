# module ID M1–M9 (same across phases) — the Wong palette extended to 9 with black and gray
const MODULE_COLORS = vcat(Makie.wong_colors(), [RGBAf(0, 0, 0, 1), RGBAf(0.6, 0.6, 0.6, 1)])

# tick label for a module index 1–27, e.g. 9 => "P1M9", 10 => "P2M1"
function module_label(i)
    p, m = divrem(Int(i) - 1, 9)
    return "P$(p + 1)M$(m + 1)"
end

function module_id_xticks!(ax)
    ax.xtickformat = values -> module_label.(values)
    ax.xticks = [1, 6, 10, 15, 19, 24]
    ax.xminorticks = 1:27
    ax.xminorticksvisible = true
    return
end
