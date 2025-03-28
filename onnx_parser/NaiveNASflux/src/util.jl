NaiveNASlib.nin(t::FluxLayer, l) = throw(ArgumentError("Not implemented for $t"))
NaiveNASlib.nout(t::FluxLayer, l) = throw(ArgumentError("Not implemented for $t"))

NaiveNASlib.nin(::FluxParLayer, l) = [size(weights(l), indim(l))]

NaiveNASlib.nin(::FluxConvolutional, l) = [size(weights(l), indim(l)) * ngroups(l)]
NaiveNASlib.nin(::FluxParInvLayer, l) = [nout(l)]

NaiveNASlib.nout(::FluxParLayer, l) = size(weights(l), outdim(l))

NaiveNASlib.nout(::FluxScale, l) = length(weights(l))
NaiveNASlib.nout(::FluxParInvLayer, l::LayerNorm) = nout(l.diag)
NaiveNASlib.nout(::FluxParNorm, l) = l.chs
NaiveNASlib.nout(::FluxRecurrent, l) = div(size(weights(l), outdim(l)), outscale(l))

outscale(l) = outscale(layertype(l))
outscale(::FluxRnn) = 1
outscale(::FluxLstm) = 4
outscale(::FluxGru) = 3

indim(l) = indim(layertype(l))
outdim(l) = outdim(layertype(l))
actdim(l) = actdim(layertype(l))
actrank(l) = actrank(layertype(l))

indim(t::FluxLayer) = throw(ArgumentError("Not implemented for $t"))
outdim(t::FluxLayer) = throw(ArgumentError("Not implemented for $t"))
actdim(t::FluxLayer) = throw(ArgumentError("Not implemented for $t"))
actrank(t::FluxLayer) = throw(ArgumentError("Not implemented for $t"))

indim(::Flux2D) = 2
outdim(::Flux2D) = 1
actdim(::Flux2D) = 1
actrank(::Flux2D) = 1

indim(::FluxScale) = 1
outdim(::FluxScale) = 1
actdim(::FluxScale) = 1
actrank(::FluxScale) = 1

indim(::FluxRecurrent) = 2
outdim(::FluxRecurrent) = 1
actdim(::FluxRecurrent) = 1
actrank(::FluxRecurrent) = 2

indim(::FluxConvolutional{N}) where N = 2+N
outdim(::FluxConvolutional{N}) where N = 1+N
actdim(::FluxConvolutional{N}) where N = 1+N
actrank(::FluxConvolutional{N}) where N = 1+N
indim(::Union{FluxConv{N}, FluxCrossCor{N}}) where N = 1+N
outdim(::Union{FluxConv{N}, FluxCrossCor{N}}) where N = 2+N
indim(::Union{FluxConvTranspose{N}, FluxCrossCor{N}}) where N = 1+N
outdim(::Union{FluxConvTranspose{N}, FluxCrossCor{N}}) where N = 2+N
# Note: Absence of bias mean that bias is a Bool (false), so beware!
weights(l) = weights(layertype(l), l)
bias(l) = bias(layertype(l), l)

weights(::FluxDense, l) = l.weight
bias(::FluxDense, l) = l.bias

weights(::FluxConvolutional, l) = l.weight
bias(::FluxConvolutional, l) = l.bias

weights(::FluxScale, l) = l.scale
bias(::FluxScale, l) = l.bias

weights(lt::FluxRecurrent, l::Flux.Recur) = weights(lt, l.cell)
bias(lt::FluxRecurrent, l::Flux.Recur) = bias(lt, l.cell)
weights(::FluxRecurrent, cell) = cell.Wi
bias(::FluxRecurrent, cell) = cell.b

hiddenweights(l) = hiddenweights(layertype(l), l)
hiddenweights(lt::FluxRecurrent, l::Flux.Recur) = hiddenweights(lt, l.cell)
hiddenweights(::FluxRecurrent, cell) = cell.Wh

hiddenstate(l) = hiddenstate(layertype(l), l)
hiddenstate(lt::FluxRecurrent, l::Flux.Recur) = hiddenstate(lt, l.cell)
hiddenstate(::FluxRecurrent, cell) = cell.state0
hiddenstate(::FluxLstm, cell::Flux.LSTMCell) = [h for h in cell.state0]

state(l) = state(layertype(l), l)
state(::FluxRecurrent, l) = l.state
state(::FluxLstm, l) = [h for h in l.state]

ngroups(l) = ngroups(layertype(l), l)
ngroups(lt, l) = 1
ngroups(lt::FluxConvolutional, l) = ngroups(lt)
ngroups(::FluxConvolutional) = 1
ngroups(lt::FluxConv) = lt.groups
ngroups(lt::FluxConvTranspose) = lt.groups
