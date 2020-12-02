# bl-home-tests

## Test 1

- Model is trained in Jupyter notebook: `Test1.ipynb`.
    - MSE after 50 epochs: **train - 0.1889, val - 0.1067. Validation split: 0.2.** 
    - Preprocessing: X* data is scaled using StandardScaler from `sklearn` library 
- Predicted values are saved in `test_pred.csv` file.

## Test 2

- Setup:
    - Ubuntu 20.04 LTS
    - RTX 2080 Ti
    - environment:
        - python 3.6
        - tensorflow 2.3.1
        - tensorRT 6.0.1
        - cuda 10.1
        - cudnn 7.6.4 
- Model conversion is implemented in `run.py`. 
    - I also left `Test2.ipynb` that contains the results of my experimental run and model benchmarking:
        - Inference in converted model runs about 2 times faster (on average over 7 runs 100 iteration each) in comparison to the original model.
- Pipeline (ref: https://www.tensorflow.org/api_docs/python/tf/experimental/tensorrt/Converter):
    - specify parameters of conversion using: `tf.experimental.tensorrt.ConversionParams(...)`
    - convert model using graph converter v2: `tf.experimental.tensorrt.TrtGraphConverterV2(...)`
    - build TensorRT engines providing function that generates data (`data_feeder()` in `run.py`) compatible with model input
    - save the result

