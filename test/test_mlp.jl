@testset "Naive MLP" begin
    function test_mlp(prop_method)
        # small_nnet encodes the simple function 24*max(x + 1.5, 0) + 18.5
        onnx_path = net_path * "small_nnet.onnx"
        # max(x+1.5, 0) max(x+1.5, 0)              [0,4]
        # 4*max(x+1.5, 0)+2.5 4*max(x+1.5, 0)+2.5  [2.5, 18.5]
        # 24*max(x+1.5, 0)+18.5                    [18.5, 114.5]
        in_hyper  = Hyperrectangle(low = [-2.5], high = [2.5]) # expected out: [18.5, 114.5]
        out_violated    = Hyperrectangle(low = [19], high = [114]) # 20.0 ≤ y ≤ 90.0
        out_holds = Hyperrectangle(low = [18], high = [115.0]) # -1.0 ≤ y ≤ 50.0
        comp_violated    = Complement(Hyperrectangle(low = [10], high = [19])) # y ≤ 10.0 or 19 ≤ y
        comp_holds    = Complement(Hyperrectangle(low = [115], high = [118])) # y ≤ 10.0 or 18 ≤ y
        info = nothing
        search_method = BFS(max_iter=100, batch_size=1)
        split_method = Bisect(1)
        
        @test verify(search_method, split_method, prop_method, Problem(onnx_path, in_hyper, out_holds)).status == :holds
        @test verify(search_method, split_method, prop_method, Problem(onnx_path, in_hyper, out_violated)).status == :violated
        @test verify(search_method, split_method, prop_method, Problem(onnx_path, in_hyper, comp_holds)).status == :holds
        @test verify(search_method, split_method, prop_method, Problem(onnx_path, in_hyper, comp_violated)).status == :violated
        
    end
    @testset "Ai2z" test_mlp(Ai2z())
    @testset "StarSet" test_mlp(StarSet())
    @testset "StarSet w/ Ai2z pre-bound" test_mlp(StarSet(Ai2z()))
    # @testset "BetaCrown GPU" test_mlp(BetaCrown(inherit_pre_bound=false)) # no gpu runner on Github Actions
    @testset "BetaCrown CPU" test_mlp(BetaCrown(use_gpu=false, inherit_pre_bound=false))
    @testset "MIPVerify" test_mlp(MIPVerify())
end
