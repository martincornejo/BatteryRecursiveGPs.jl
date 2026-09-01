function module_id_xticks!(ax)
    ax.xtickformat = values -> begin
        map(values) do value
            p, m = divrem(Int(value), 9)
            "P$(p + 1)M$m"
        end
    end
    ax.xticks = [1, 6, 10, 15, 19, 24]
    ax.xminorticks = 1:27
    ax.xminorticksvisible = true
    return
end
