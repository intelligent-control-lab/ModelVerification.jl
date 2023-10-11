using Documenter, ModelVerification

makedocs(sitename = "ModelVerification.jl",
         pages = ["index.md", "problem.md", "solvers.md", "functions.md", "existing_implementations.md"])


deploydocs(
    repo = "github.com/intelligent-control-lab/ModelVerification.jl.git",
)