using Optimisers


function opt_step(opt_state, tpostr, xr)
    grad, val = gl(tpostr, xr)
    Optimisers.update!(opt_state, xr, -grad), -val
end



function callback(tpost, xr)
    ps = Comrade.transform(tpost, xr);
    ms = skymodel(tpost.lpost, ps)
    res = residuals(tpost.lpost, ps)
    return ps, ms, res
end
cb = @compile callback(tpost, xr)

xr = Reactant.to_rarray(prior_sample(tpost))
rule = Optimisers.AdamW(0.01)
opt_state = @jit Optimisers.setup(rule, xr)
os = @compile opt_step(opt_state, tpost, xr)

using Adapt, CairoMakie
gpl = refinespatial(grd, 2)


_, val0 =os(opt_state, tpost, xr)
for i in 1:1000
    _, val = os(opt_state, tpost, xr)
    if i % 100 == 0 || i == 1
        @info i val
        ps, ms, res = cb(tpost, xr)
        img0 = adapt(Array, parent(VLBISkyModels.unmodified(ms)))
        fig = imageviz(img0, colormap = :cmr_gothic, pcolormap = :rainbow1, plot_total = false)
        # res1 = Adapt.adapt(Array, res[1])
        # dt = datatable(res1)[1:10:end]
        # uvdv = uvdist.(dt)
        # r11 = measurement.(dt)
        # n11 = noise.(dt)
        # ax, pl = scatter(fig[1, 2], uvdv, real.(r11.:1 ./ n11.:1), axis = (; ylabel = "RR Residual"))
        # scatter!(ax, uvdv, imag.(r11.:1 ./ n11.:1))
        # colsize!(fig.layout, 2, Relative(0.6))
        display(fig)
        GC.gc()
    end
end
