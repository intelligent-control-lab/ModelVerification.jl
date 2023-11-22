using Documenter, ModelVerification
# push!(LOAD_PATH,"../src/")

makedocs(sitename = "ModelVerification.jl",
        #  modules = [ModelVerification],
         pages = [
            "index.md", 
            "Toolbox Outline" => [
                "Problem" => [
                    "problem.md",
                    "network.md",
                    "safety_spec.md",
                ],
                "branching.md",
                "propagate.md",
                "solvers.md",
                "attack.md",
                "utils.md"
            ],
            "Python Interface" => [
                "nnet_converter.md",
                "python_interface.md"
            ],
            "Benchmarks " => [
                "benchmark.md"
            ],
            "existing_implementations.md", 
            "about.md"
        ]
)
