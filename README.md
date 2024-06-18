# ModelVerification.jl

For an extensive documentation of the toolbox, please run a local web server out of the `docs/build` directory. 

First, install the [LiveServer](https://github.com/tlienart/LiveServer.jl) Julia package.
```Julia
$ julia
julia> import Pkg
julia> Pkg.add("LiveServer")
```

Then, start the server with one of the following methods:
1. In Julia REPL
```Julia
julia> using LiveServer
julia> serve(dir="docs/build")
```
2. In console, using Julia
```console
julia -e 'using LiveServer; serve(dir="docs/build")'
```

This should take you to a full documentation of the toolbox. For more information, please follow the [Note in "Building an Empty Document" section](https://documenter.juliadocs.org/stable/man/guide/#Building-an-Empty-Document).

An alternate is to click on the `index.html` file in the `docs/build` folder.