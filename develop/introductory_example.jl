
using ModelVerification, LazySets
const MV = ModelVerification

onnx_path = "models/small_nnet.onnx"

toy_model = MV.build_flux_model(onnx_path)

X = Hyperrectangle(low = [-2.5], high = [2.5])
Y = Hyperrectangle(low = [18.5], high = [114.5])
problem = Problem(toy_model, X, Y, onnx_path)

search_method = BFS(max_iter=100, batch_size=1)
split_method = Bisect(1)

solver = Crown(use_gpu=false, bound_lower=true, bound_upper=true)

result = verify(search_method, split_method, solver, problem)
