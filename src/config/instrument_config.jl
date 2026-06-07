# The instrument config is fully handled by the generic assembler; this is the named entry
# point that mirrors the other `build_*_config` functions.

"""
    build_instrument_config(cfg::AbstractDict) -> InstrumentModel

Build the instrument model from a parsed instrument TOML. See [`assemble_instrument`](@ref).
"""
build_instrument_config(cfg::AbstractDict) = assemble_instrument(cfg)
