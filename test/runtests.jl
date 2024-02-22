using ModelVerification, LazySets, Flux
using Test

macro no_error(ex)
    quote
        try $(esc(ex))
            true
        catch e
            @error(e)
            false
        end
    end
end

net_path = joinpath(@__DIR__, "networks/")

include("test_mlp.jl")