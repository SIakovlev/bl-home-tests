import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import numpy as np
import time
import argparse

# Create some dummy data which we use for benchmarking and conversion
# (info about input size was taken from pipeline.config file)
data = tf.constant(np.empty([1, 640, 640, 3], np.uint8))


def benchmark_model(path, N=100, custom_op=False):

    model = tf.saved_model.load(path).signatures["serving_default"] if custom_op \
        else tf.saved_model.load(path)

    print(f"Run speed benchmark for the following model: {path}")

    start = time.time()
    for i in range(N):
        model(data)
    end = time.time()

    print(f"Result: time elapsed: {end - start}")


def data_feeder():
    yield [data]


def run(**kwargs):
    input_model_dir = kwargs['input_model_dir']
    output_model_dir = kwargs['output_model_dir']
    run_benchmark = kwargs['run_benchmark']
    run_conversion = kwargs['run_conversion']

    if run_conversion:
        params = tf.experimental.tensorrt.ConversionParams(precision_mode='FP16')
        converter = trt.TrtGraphConverterV2(input_saved_model_dir=input_model_dir, conversion_params=params)
        converter.convert()
        converter.build(input_fn=data_feeder)
        converter.save(output_model_dir)

    if run_benchmark:
        benchmark_model(input_model_dir)
        benchmark_model(output_model_dir, custom_op=True)


if __name__ == '__main__':

    # seems that linking correct cudnn7.6.4 solves this issue...
    # gpu = tf.config.experimental.list_physical_devices('GPU')
    # tf.config.experimental.set_memory_growth(gpu[0], True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_model_dir',
                        type=str,
                        default='./Exported_model_V1/saved_model/')
    parser.add_argument('--output_model_dir',
                        type=str,
                        default='./Exported_model_V1/converted_model/')
    parser.add_argument('--run_benchmark',
                        type=bool,
                        default=True)
    parser.add_argument('--run_conversion',
                        type=bool,
                        default=True)

    args = parser.parse_args()

    run(**vars(args))

