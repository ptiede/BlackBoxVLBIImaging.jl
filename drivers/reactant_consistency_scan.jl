# Reactant device-vs-CPU consistency scan.
#
# Motivation: the Reactant fit (optimization AND sampling) produces garbage for every sky
# model, yet a log-density/gradient check at a *prior draw* passes. Optimization and sampling
# share only the device posterior (`prepare_device`) + its `asflat` transform, so a cause that
# breaks both must live there. The prior draw has gains ~ 1 and leakage d-terms ~ 0, so it
# never probes the parameter region the fit actually reaches.
#
# This script compares the CPU/Enzyme and Reactant-device posteriors NOT at a prior draw but
# along the directions the fit moves in — scaling the gain/leakage block up toward physical
# values — and reports the divergence PER PARAMETER BLOCK (sky vs gain vs leakage), with a
# block-local relative tolerance (the production check normalizes the gradient by the global
# max, which is sky-dominated and hides an O(1) instrument-block error).
#
# Usage (own driver env; -t auto to match a real run's executor/threads):
#   julia --project=drivers -t auto drivers/reactant_consistency_scan.jl \
#       [image.toml instrument.toml data.toml fitting.toml] [optimum.jls]
#
# With no args it defaults to examples/2024_3c279. A 5th arg is an optional serialized
# optimum (`Dict(:xopt=>...)` from a run) to use as the base point instead of a prior draw —
# pass a CPU-converged optimum to see the divergence exactly where the fit lands.

using BlackBoxVLBIImaging
using Random, Printf, TOML

const BBI = BlackBoxVLBIImaging
const Reactant = BBI.Reactant
const Optimisers = BBI.Optimisers
const Comrade = BBI.Comrade
const LDP = BBI.LogDensityProblems

# ---- args / defaults ----------------------------------------------------------------------
const EXDIR = joinpath(@__DIR__, "..", "examples", "2024_3c279")
imgf, intf, datf, fitf = if length(ARGS) >= 4
    ARGS[1], ARGS[2], ARGS[3], ARGS[4]
else
    joinpath(EXDIR, "image.toml"), joinpath(EXDIR, "instrument.toml"),
    joinpath(EXDIR, "data.toml"), joinpath(EXDIR, "fitting_reactant.toml")
end
optjls = length(ARGS) >= 5 ? ARGS[5] : nothing
const RTOL = 1.0e-2

# ---- build the SAME objects the pipeline builds -------------------------------------------
@info "Building configs" image = imgf instrument = intf data = datf fitting = fitf
skycfg = TOML.parsefile(imgf); intcfg = TOML.parsefile(intf)
datacfg = TOML.parsefile(datf); fitcfg = TOML.parsefile(fitf)

strategy = build_fitting_config(fitcfg)
(skym, imgdata) = build_sky_config(skycfg)
intm = build_instrument_config(intcfg)
dcoh = build_data_config(datacfg)

# CPU/Enzyme posterior (the reference) and the Reactant device posterior — exactly what
# `comrade_imager` and `reactant_opt` construct.
post = VLBIPosterior(skym, intm, dcoh; imgdata)
@info "Moving posterior to the Reactant device (prepare_device)"
dpost = Comrade.prepare_device(post, Comrade.ComradeBase.ReactantEx())
tc = asflat(post)     # CPU transform
td = asflat(dpost)    # device transform (must be structurally identical to tc)

# ---- base point: prior draw (or a supplied optimum) ---------------------------------------
rng = Random.seed!(Random.default_rng(), 42)
p0 = if isnothing(optjls)
    @info "Base point: prior draw (gains~1, leakage~0)"
    prior_sample(rng, post)
else
    @info "Base point: supplied optimum" file = optjls
    BBI.Serialization.deserialize(optjls)[:xopt]
end
x0 = Comrade.inverse(tc, p0)
dim = length(x0)
@info "Flat dimension" dim

# ---- identify the flat-index blocks (sky / gain / leakage) --------------------------------
# Perturb a group of params in constrained space and see which flat coords move. This finds
# each block's flat slice without hard-coding the posterior's internal ordering.
function _changed(p1)
    x1 = Comrade.inverse(tc, p1)
    return findall(i -> abs(x0[i] - x1[i]) > 1.0e-9, eachindex(x0))
end
_isleak(k) = (s = String(k); startswith(s, "d") && !endswith(s, "μ") && !endswith(s, "σ"))

inst_bump(δ) = merge(p0, (; instrument = map(v -> v .+ δ, p0.instrument)))
function leak_bump(δ)
    inst = p0.instrument; ks = keys(inst)
    vals = map(k -> _isleak(k) ? inst[k] .+ δ : inst[k], ks)
    return merge(p0, (; instrument = NamedTuple{ks}(vals)))
end

Iinst = _changed(inst_bump(0.01))
Ileak = _changed(leak_bump(0.01))
Igain = setdiff(Iinst, Ileak)
Isky = setdiff(1:dim, Iinst)
@info "Blocks (flat indices)" nsky = length(Isky) ngain = length(Igain) nleak = length(Ileak)

# ---- device value+gradient via the production step (Descent(1) trick) ---------------------
# Reuse `_reactant_opt_step!` with a unit Descent so one compiled call gives:
#   val   = -logdensity(device)         (the step's loss)
#   new_x = x + ∇logdensity(device)     (Descent(1): x - ∇(-logdensity))
# i.e. device gradient = new_x - x. This is the EXACT compiled path the optimizer uses, and
# avoids the value+grad top-level @compile that overflows Julia inference.
xr = Reactant.to_rarray(copy(x0))
st = Reactant.@jit Optimisers.setup(Optimisers.Descent(1), xr)
@info "Compiling device step (one-time)"
step = Reactant.@compile BBI._reactant_opt_step!(st, td, xr)

function device_vg(x)
    copyto!(xr, x)
    _, nx, val = step(st, td, xr)
    ld = -Float64(Reactant.@allowscalar val)
    g = collect(Comrade.Adapt.adapt(Array, nx)) .- x
    return ld, g
end
function cpu_vg(x)
    vc, gc = LDP.logdensity_and_gradient(tc, x)
    return vc, collect(gc)
end

# block-local max relative gradient error, normalized within the block (not by the global max)
function blockrel(gc, gd, I)
    isempty(I) && return 0.0
    return maximum(abs.(gc[I] .- gd[I])) / max(maximum(abs, @view gc[I]), eps())
end

# largest constrained |d-term| at a flat point (interpretability for the leakage scan)
function leak_maxmag(x)
    inst = Comrade.transform(tc, x).instrument
    m = 0.0
    if haskey(inst, :d1re) && haskey(inst, :d1im)
        m = max(m, maximum(sqrt.(abs2.(inst.d1re) .+ abs2.(inst.d1im))))
    end
    if haskey(inst, :d2re) && haskey(inst, :d2im)
        m = max(m, maximum(sqrt.(abs2.(inst.d2re) .+ abs2.(inst.d2im))))
    end
    return m
end

# ---- sanity at the base point (this is the check the user already "passed") ---------------
vc0, gc0 = cpu_vg(x0)
vd0, gd0 = device_vg(x0)
@info "Base-point check (should agree — reproduces the passing prior-draw check)" cpu_ld = vc0 device_ld = vd0 val_reldiff = abs(vc0 - vd0) / max(abs(vc0), 1) grad_reldiff_global = maximum(abs.(gc0 .- gd0)) / max(maximum(abs, gc0), eps())

println()
println("="^96)
println("Per-block gradient reldiff at the base point (global-normalized check would MISS these):")
@printf("  sky:  %.3e   gain: %.3e   leak: %.3e\n",
    blockrel(gc0, gd0, Isky), blockrel(gc0, gd0, Igain), blockrel(gc0, gd0, Ileak))
println("="^96)

# ---- scan 1: leakage toward physical magnitudes -------------------------------------------
function run_scan(title, header, points, makex)
    println("\n", title)
    println(header)
    first_div = nothing
    for s in points
        x = makex(s)
        vc, gc = cpu_vg(x)
        vd, gd = device_vg(x)
        vrel = abs(vc - vd) / max(abs(vc), 1)
        gl = blockrel(gc, gd, Ileak); gg = blockrel(gc, gd, Igain); gs = blockrel(gc, gd, Isky)
        dmax = leak_maxmag(x)
        flag = (vrel > RTOL || gl > RTOL || gg > RTOL) ? "  <-- DIVERGES" : ""
        @printf("  s=%6.3f |d|max=%5.3f  ld_cpu=%+.6e ld_dev=%+.6e  vrel=%.2e  g[sky]=%.2e g[gain]=%.2e g[leak]=%.2e%s\n",
            s, dmax, vc, vd, vrel, gs, gg, gl, flag)
        if isnothing(first_div) && !isempty(flag)
            first_div = (; s, dmax, vrel, gl, gg)
        end
    end
    return first_div
end

div1 = nothing
if !isempty(Ileak)
    # Set every leakage unconstrained coord to s: |d-term| grows ~ s. 0 = no leakage.
    div1 = run_scan(
        "SCAN 1 — leakage block set to magnitude s (sky+gain held at base point):",
        "  (CPU vs device along the direction the fit pushes the d-terms)",
        0.0:0.025:0.4,
        s -> (x = copy(x0); x[Ileak] .= s; x),
    )
else
    @info "No leakage block found (leakage scheme = none); skipping SCAN 1."
end

# ---- scan 2: scale the whole instrument block away from the base point ---------------------
# Generic probe (works with any gain/leakage scheme): x_inst = k * x0_inst.
div2 = run_scan(
    "SCAN 2 — instrument block scaled by factor k (sky held at base point):",
    "  (k=1 is the base point; k>1 pushes gains+leakage proportionally further)",
    0.0:0.5:8.0,
    k -> (x = copy(x0); x[Iinst] .= k .* x0[Iinst]; x),
)

# ---- verdict ------------------------------------------------------------------------------
println("\n", "="^96)
if isnothing(div1) && isnothing(div2)
    println("No divergence found up to the scanned range (rtol=$RTOL).")
    println("The device posterior tracks CPU even at large instrument values — the bug is NOT")
    println("a device-likelihood discrepancy. Look at the transform/initial-point handoff instead.")
else
    println("DIVERGENCE FOUND — the device log-density/gradient peels away from CPU as the")
    println("instrument terms grow, while matching at the base point. Both the optimizer and the")
    println("sampler then target a distorted posterior -> garbage, for every sky model.")
    !isnothing(div1) && @printf("  leakage scan: first diverges at s=%.3f (|d|max=%.3f): vrel=%.2e g[leak]=%.2e\n",
        div1.s, div1.dmax, div1.vrel, div1.gl)
    !isnothing(div2) && @printf("  instrument scan: first diverges at k=%.3f: vrel=%.2e g[gain]=%.2e\n",
        div2.s, div2.vrel, div2.gg)
    println("\nNext: bisect the instrument Jones path (JonesSandwich g*d*r / field-rotation")
    println("JonesR / refbasis / corr_polbasis) under Reactant at the diverging point.")
end
println("="^96)
