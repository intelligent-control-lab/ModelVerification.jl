import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import warnings; warnings.filterwarnings("ignore")
os.environ['TF_KERAS'] = '1'
import tensorflow as tf
import argparse
from math import isclose
import onnxmltools
import onnx
import onnxruntime as rt


def convert_modelh5_to_onnx(path_h5, name_h5, test_conversion=False):
    
    model_tf = tf.keras.models.load_model( f"{path_h5}{name_h5}.h5", compile=False)
    #model_tf.summary()
    onnx_model = onnxmltools.convert_keras(model_tf) 
    onnxmltools.utils.save_model(onnx_model, f'{name_h5}.onnx')

    #TEST CONVERSION
    if test_conversion:
        import numpy as np
        cloud_size = 1
        input_area = np.array([[0, 1]]*model_tf.layers[0].input_shape[0][1])
        input_area = input_area.reshape(1, input_area.shape[0], 2)
        domains = np.array([np.random.uniform(i[:, 0], i[:, 1], size=(cloud_size, input_area.shape[1])) for i in input_area])
        network_input = domains.reshape( cloud_size*input_area.shape[0], -1 )
        res_tf = model_tf(network_input)[0]

        #onnx preparation
        session = rt.InferenceSession(f'{name_h5}.onnx')
        
        # Run forward propagation onnx
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        # Perform inference
        res_onnx = session.run([output_name], {input_name: network_input.astype("float32")})[0][0]

        
        print(f'\nResult forward propagation in model.h5: {res_tf}') 
        print(f'\nResult forward propagation in model.onnx: {res_onnx}\n') 

        if np.allclose(res_tf,res_onnx):
            print("\t====> The models are the same! Conversion succefully done!\n")
        else:
            print('\tX ===> The models are not the same! Something went wrong...')
            
    

if __name__ == '__main__':
    # Read user inputs and run nnet2onnx function for different numbers of inputs
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="./", help='Path where model should be loaded/saved.')
    parser.add_argument('--name_model', type=str, default="model_test", help='Path where model should be loaded/saved.')
    parser.add_argument('--test_conversion', type=bool, default=True, help='Whether to test the conversion')
    args = parser.parse_args()

    print('Starting conversions...')
    convert_modelh5_to_onnx(args.model_path, args.name_model, args.test_conversion)