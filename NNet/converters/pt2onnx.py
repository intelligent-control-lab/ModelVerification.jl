import torch
import argparse
from math import isclose
import onnxmltools
import onnx
import onnxruntime as rt
import numpy as np


class Net(torch.nn.Module):
	def __init__(self, input_size, output_size):
		super(Net, self).__init__()
		self.fc1 = torch.nn.Linear(input_size, 32)
		self.fc2 = torch.nn.Linear(32, 32)
		self.fc3 = torch.nn.Linear(32, output_size)

	def forward(self, x):
		x = torch.torch.nn.functional.relu(self.fc1(x))
		x = torch.torch.nn.functional.relu(self.fc2(x))
		x = self.fc3(x)
		return x


def convert_modelpt_to_onnx(path_pt, name_pt, test_conversion=False):
    
    model_pt = Net(2,1)
    model_pt.load_state_dict(torch.load(f"{path_pt}{name_pt}"))
    model_pt.eval()
   
    onnx_path = "model_test_pt.onnx"   
    dummy_input = torch.randn(1, list(model_pt.parameters())[0].shape[1])
    torch.onnx.export(model_pt, dummy_input, onnx_path, verbose=True)
    
    #TEST CONVERSION
    if test_conversion:
      
        res_pt = model_pt(dummy_input).detach().numpy()[0]

        #onnx preparation
        session = rt.InferenceSession(f'{onnx_path}')
        
        # Run forward propagation onnx
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        # Perform inference
        res_onnx = session.run([output_name], {input_name: dummy_input.detach().numpy()})[0][0]

        
        print(f'\nResult forward propagation in model.h5: {res_pt}') 
        print(f'\nResult forward propagation in model.onnx: {res_onnx}\n') 

        if np.allclose(res_pt,res_onnx):
            print("\t====> The models are the same! Conversion succefully done!\n")
        else:
            print('\tX ===> The models are not the same! Something went wrong...')
            
    

if __name__ == '__main__':
    # Read user inputs and run nnet2onnx function for different numbers of inputs
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="./", help='Path where model should be loaded/saved.')
    parser.add_argument('--name_model', type=str, default="model_test.pt", help='Path where model should be loaded/saved.')
    parser.add_argument('--test_conversion', type=bool, default=True, help='Whether to test the conversion')
    args = parser.parse_args()

    print('Starting conversions...')
    convert_modelpt_to_onnx(args.model_path, args.name_model, args.test_conversion)