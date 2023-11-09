using Documenter, ModelVerification
push!(LOAD_PATH,"../src/")

makedocs(sitename = "ModelVerification.jl",
         pages = ["index.md", "existing_implementations.md"])


# deploydocs(
#     repo = "github.com/intelligent-control-lab/ModelVerification.jl.git",
# )