using Documenter, ModelVerification
# push!(LOAD_PATH,"../src/")

makedocs(sitename = "ModelVerification.jl",
         format = Documenter.HTML(prettyurls = false),
         pages = [
            "index.md", 
            "Toolbox Outline" => [
                "toolbox_flow.md",
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
            "existing_implementations.md", 
            "about.md"
        ]
)

deploydocs(; repo="github.com/intelligent-control-lab/ModelVerification.jl.git",
           push_preview=true)