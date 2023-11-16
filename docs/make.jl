using Documenter, ModelVerification
push!(LOAD_PATH,"../src/")

makedocs(sitename = "ModelVerification.jl",
         pages = [
            "index.md", 
            "Package Outline" => [
                "problem.md", 
                "solvers.md"
            ],
            "existing_implementations.md", 
            "about.md"
        ]
)
