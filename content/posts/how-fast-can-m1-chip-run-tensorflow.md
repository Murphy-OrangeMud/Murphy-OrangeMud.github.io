---
title: "How Fast Can M1 Chip Run Tensorflow"
date: 2022-10-07T15:06:53-04:00
draft: false
tags: ["M1 Mac"]
---

Having heard of the amazing computing power of M1-chip long before, I can't miss a chance to try on my M1 Mac (2020 Mbp with M1 Chip) in excitement. Because of my little experience on Tensorflow, I compared using some example codes and didn't carry out large-scale benchmarks.

### Install Tensorflow on M1 Mac
As a friend of bugs who fought with Python for hundreds of times, I still stepped into trouble during my installation. At first I used conda and reinstalled several times but after each installation I met the problem:

```
>>> import tensorflow as tf
zsh: illegal hardware instruction  python
```

After searching on the Internet, I found that it is because that Tensorflow doesn't support MacOS but only Ubuntu and Windows. Okay then I'll install tensorflow-macos, but another problem occurred:

```
 (base) ~ pip install tensorflow-macos
ERROR: Could not find a version that satisfies the requirement tensorflow-macos (from versions: none)
ERROR: No matching distribution found for tensorflow-macos
```

After searching on the Internet hard for the whole morning, I found the solution. Many said that we should install miniforge for arm64 and install tensorflow-macos in this kind of "conda" (Refer [this post](https://www.jianshu.com/p/7d27a53e3a5e)) but I failed again.

OK so my solution is as follows:
My hardware version: Apple M1-chip, OS X 12.1 Monterey

First we shall download and install python 3.9.4 universal 64bit version from python.org (I can't find the original solution website) and then run (Refer [this quetion](https://developer.apple.com/forums/thread/683757)):

``` bash
$ python3 -m venv tensorflow-metal-test
$ source tensorflow-metal-test/bin/activate
$ cd tensorflow-metal-test/
$ python -m pip install -U pip
$ pip install tensorflow-macos
$ pip install tensorflow-metal
```

But it will raise error:

```
Building wheels for collected packages: h5py
  Building wheel for h5py (pyproject.toml) ... error
  error: subprocess-exited-with-error
  
  × Building wheel for h5py (pyproject.toml) did not run successfully.
  │ exit code: 1
  ╰─> [70 lines of output]
      running bdist_wheel
      running build
      running build_py
      creating build
      creating build/lib.macosx-10.9-universal2-3.9
      creating build/lib.macosx-10.9-universal2-3.9/h5py
      copying h5py/h5py_warnings.py -> build/lib.macosx-10.9-universal2-3.9/h5py
      copying h5py/version.py -> build/lib.macosx-10.9-universal2-3.9/h5py
      copying h5py/__init__.py -> build/lib.macosx-10.9-universal2-3.9/h5py
      copying h5py/ipy_completer.py -> build/lib.macosx-10.9-universal2-3.9/h5py
      creating build/lib.macosx-10.9-universal2-3.9/h5py/_hl
      copying h5py/_hl/files.py -> build/lib.macosx-10.9-universal2-3.9/h5py/_hl
      copying h5py/_hl/compat.py -> build/lib.macosx-10.9-universal2-3.9/h5py/_hl
      copying h5py/_hl/__init__.py -> build/lib.macosx-10.9-universal2-3.9/h5py/_hl
      copying h5py/_hl/selections.py -> build/lib.macosx-10.9-universal2-3.9/h5py/_hl
      copying h5py/_hl/dataset.py -> build/lib.macosx-10.9-universal2-3.9/h5py/_hl
      copying h5py/_hl/vds.py -> build/lib.macosx-10.9-universal2-3.9/h5py/_hl
      copying h5py/_hl/selections2.py -> build/lib.macosx-10.9-universal2-3.9/h5py/_hl
      copying h5py/_hl/group.py -> build/lib.macosx-10.9-universal2-3.9/h5py/_hl
      copying h5py/_hl/datatype.py -> build/lib.macosx-10.9-universal2-3.9/h5py/_hl
      copying h5py/_hl/attrs.py -> build/lib.macosx-10.9-universal2-3.9/h5py/_hl
      copying h5py/_hl/dims.py -> build/lib.macosx-10.9-universal2-3.9/h5py/_hl
      copying h5py/_hl/base.py -> build/lib.macosx-10.9-universal2-3.9/h5py/_hl
      copying h5py/_hl/filters.py -> build/lib.macosx-10.9-universal2-3.9/h5py/_hl
      creating build/lib.macosx-10.9-universal2-3.9/h5py/tests
      copying h5py/tests/test_dimension_scales.py -> build/lib.macosx-10.9-universal2-3.9/h5py/tests
      copying h5py/tests/test_attribute_create.py -> build/lib.macosx-10.9-universal2-3.9/h5py/tests
      copying h5py/tests/test_file_image.py -> build/lib.macosx-10.9-universal2-3.9/h5py/tests
      copying h5py/tests/conftest.py -> build/lib.macosx-10.9-universal2-3.9/h5py/tests
      copying h5py/tests/test_h5d_direct_chunk.py -> build/lib.macosx-10.9-universal2-3.9/h5py/tests
      copying h5py/tests/test_h5f.py -> build/lib.macosx-10.9-universal2-3.9/h5py/tests
      copying h5py/tests/test_dataset_getitem.py -> build/lib.macosx-10.9-universal2-3.9/h5py/tests
      copying h5py/tests/test_group.py -> build/lib.macosx-10.9-universal2-3.9/h5py/tests
      copying h5py/tests/test_errors.py -> build/lib.macosx-10.9-universal2-3.9/h5py/tests
      copying h5py/tests/test_dataset_swmr.py -> build/lib.macosx-10.9-universal2-3.9/h5py/tests
      copying h5py/tests/test_slicing.py -> build/lib.macosx-10.9-universal2-3.9/h5py/tests
      copying h5py/tests/test_h5pl.py -> build/lib.macosx-10.9-universal2-3.9/h5py/tests
      copying h5py/tests/test_attrs.py -> build/lib.macosx-10.9-universal2-3.9/h5py/tests
      copying h5py/tests/__init__.py -> build/lib.macosx-10.9-universal2-3.9/h5py/tests
      copying h5py/tests/test_attrs_data.py -> build/lib.macosx-10.9-universal2-3.9/h5py/tests
      copying h5py/tests/test_h5t.py -> build/lib.macosx-10.9-universal2-3.9/h5py/tests
      copying h5py/tests/test_big_endian_file.py -> build/lib.macosx-10.9-universal2-3.9/h5py/tests
      copying h5py/tests/test_h5p.py -> build/lib.macosx-10.9-universal2-3.9/h5py/tests
      copying h5py/tests/test_dims_dimensionproxy.py -> build/lib.macosx-10.9-universal2-3.9/h5py/tests
      copying h5py/tests/test_h5o.py -> build/lib.macosx-10.9-universal2-3.9/h5py/tests
      copying h5py/tests/test_datatype.py -> build/lib.macosx-10.9-universal2-3.9/h5py/tests
      copying h5py/tests/common.py -> build/lib.macosx-10.9-universal2-3.9/h5py/tests
      copying h5py/tests/test_dataset.py -> build/lib.macosx-10.9-universal2-3.9/h5py/tests
      copying h5py/tests/test_file.py -> build/lib.macosx-10.9-universal2-3.9/h5py/tests
      copying h5py/tests/test_selections.py -> build/lib.macosx-10.9-universal2-3.9/h5py/tests
      copying h5py/tests/test_dtype.py -> build/lib.macosx-10.9-universal2-3.9/h5py/tests
      copying h5py/tests/test_h5.py -> build/lib.macosx-10.9-universal2-3.9/h5py/tests
      copying h5py/tests/test_file2.py -> build/lib.macosx-10.9-universal2-3.9/h5py/tests
      copying h5py/tests/test_completions.py -> build/lib.macosx-10.9-universal2-3.9/h5py/tests
      copying h5py/tests/test_filters.py -> build/lib.macosx-10.9-universal2-3.9/h5py/tests
      copying h5py/tests/test_base.py -> build/lib.macosx-10.9-universal2-3.9/h5py/tests
      copying h5py/tests/test_objects.py -> build/lib.macosx-10.9-universal2-3.9/h5py/tests
      creating build/lib.macosx-10.9-universal2-3.9/h5py/tests/data_files
      copying h5py/tests/data_files/__init__.py -> build/lib.macosx-10.9-universal2-3.9/h5py/tests/data_files
      creating build/lib.macosx-10.9-universal2-3.9/h5py/tests/test_vds
      copying h5py/tests/test_vds/test_highlevel_vds.py -> build/lib.macosx-10.9-universal2-3.9/h5py/tests/test_vds
      copying h5py/tests/test_vds/test_virtual_source.py -> build/lib.macosx-10.9-universal2-3.9/h5py/tests/test_vds
      copying h5py/tests/test_vds/__init__.py -> build/lib.macosx-10.9-universal2-3.9/h5py/tests/test_vds
      copying h5py/tests/test_vds/test_lowlevel_vds.py -> build/lib.macosx-10.9-universal2-3.9/h5py/tests/test_vds
      copying h5py/tests/data_files/vlen_string_s390x.h5 -> build/lib.macosx-10.9-universal2-3.9/h5py/tests/data_files
      copying h5py/tests/data_files/vlen_string_dset_utc.h5 -> build/lib.macosx-10.9-universal2-3.9/h5py/tests/data_files
      copying h5py/tests/data_files/vlen_string_dset.h5 -> build/lib.macosx-10.9-universal2-3.9/h5py/tests/data_files
      running build_ext
      Building h5py requires pkg-config unless the HDF5 path is explicitly specified
      error: pkg-config probably not installed: FileNotFoundError(2, 'No such file or directory')
      [end of output]
  
  note: This error originates from a subprocess, and is likely not a problem with pip.
  ERROR: Failed building wheel for h5py
Failed to build h5py
ERROR: Could not build wheels for h5py, which is required to install pyproject.toml-based projects
```

We must install hdf5 using homebrew first:

``` bash
arch -arm64 brew install hdf5
```

(So slow...)

But it failed again. We should run (Refer [this post](https://www.logcg.com/en/archives/3548.html)):

``` bash
find /opt -iname "*hdf5.h*"
```

We can find:
```
/opt/homebrew/include/hdf5.h
/opt/homebrew/Cellar/hdf5/1.13.0/include/hdf5.h
```

And we set the environment variables:
``` bash
export CPATH="/opt/homebrew/include/"
export HDF5_DIR=/opt/homebrew/
```

And we run:
``` bash
$ pip install tensorflow-macos
$ pip install tensorflow-metal
```

We run python again and after a long time we can see:
```
Python 3.9.4 (v3.9.4:1f2e3088f3, Apr  4 2021, 12:19:19) 
[Clang 12.0.0 (clang-1200.0.32.29)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import tensorflow
 
>>> 
```

Yes! And the code is as follows.

### The Experiment

First let's see whether we can see M1 GPU in tf's devices:
```
>>> import tensorflow as tf
>>> tf.config.experimental.list_physical_devices()
[PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU'), PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

Wow!

``` python
# env: tf-test

import tensorflow as tf
from tensorflow import keras
import ssl

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

ssl._create_default_https_context = ssl._create_unverified_context

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

y_train = keras.utils.to_categorical(y_train, num_classes=10, dtype='float32')
y_test = keras.utils.to_categorical(y_test, num_classes=10, dtype='float32')

# from tensorflow.python.compiler.mlcompute import mlcompute
# mlcompute.set_mlc_device(device_name="gpu")

with tf.device('/GPU:0'):
        model = keras.Sequential([keras.layers.Flatten(input_shape=(32,32,3)),
                                keras.layers.Dense(3000, activation='relu'),
                                keras.layers.Dense(1000, activation='relu'),
                                keras.layers.Dense(10, activation='sigmoid')
                        ])

        model.compile(optimizer="SGD", loss="categorical_crossentropy", metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=5)
        model.evaluate(x_test, y_test, verbose=2)
```

The references for these code are:

https://zhuanlan.zhihu.com/p/463931687

https://www.tensorflow.org/tutorials/quickstart/beginner

https://www.analyticsvidhya.com/blog/2021/11/benchmarking-cpu-and-gpu-performance-with-tensorflow/

https://tensorflow.juejin.im/community/benchmarks.html

https://stackoverflow.com/questions/69687794/unable-to-manually-load-cifar10-dataset


### The Result


Using M1 GPU:
```
Train on 50000 samples
Metal device set to: Apple M1

systemMemory: 16.00 GB
maxCacheSize: 5.33 GB

2022-03-14 11:32:41.119904: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:305] Could not identify NUMA node of platform GPU ID 0, defaulting to 0. Your kernel may not have been built with NUMA support.
2022-03-14 11:32:41.120025: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:271] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 0 MB memory) -> physical PluggableDevice (device: 0, name: METAL, pci bus id: <undefined>)
2022-03-14 11:32:41.125025: W tensorflow/core/platform/profile_utils/cpu_utils.cc:128] Failed to get CPU frequency: 0 Hz
2022-03-14 11:32:41.125164: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:113] Plugin optimizer for device_type GPU is enabled.
2022-03-14 11:32:41.133741: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:113] Plugin optimizer for device_type GPU is enabled.
2022-03-14 11:32:41.449048: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:113] Plugin optimizer for device_type GPU is enabled.
Epoch 1/5
2022-03-14 11:32:41.453886: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:113] Plugin optimizer for device_type GPU is enabled.
50000/50000 [==============================] - 17s 331us/sample - loss: 1.8126 - accuracy: 0.3534
Epoch 2/5
50000/50000 [==============================] - 15s 291us/sample - loss: 1.6244 - accuracy: 0.4282
Epoch 3/5
50000/50000 [==============================] - 15s 292us/sample - loss: 1.5423 - accuracy: 0.4572
Epoch 4/5
50000/50000 [==============================] - 15s 292us/sample - loss: 1.4819 - accuracy: 0.4761
Epoch 5/5
50000/50000 [==============================] - 15s 293us/sample - loss: 1.4323 - accuracy: 0.4953
```

Using M1 CPU:
```
Train on 50000 samples
Metal device set to: Apple M1

systemMemory: 16.00 GB
maxCacheSize: 5.33 GB

2022-03-14 11:38:11.071008: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:305] Could not identify NUMA node of platform GPU ID 0, defaulting to 0. Your kernel may not have been built with NUMA support.
2022-03-14 11:38:11.071162: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:271] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 0 MB memory) -> physical PluggableDevice (device: 0, name: METAL, pci bus id: <undefined>)
2022-03-14 11:38:11.075712: W tensorflow/core/platform/profile_utils/cpu_utils.cc:128] Failed to get CPU frequency: 0 Hz
Epoch 1/5
50000/50000 [==============================] - 24s 477us/sample - loss: 1.8087 - accuracy: 0.3563
Epoch 2/5
50000/50000 [==============================] - 24s 489us/sample - loss: 1.6201 - accuracy: 0.4260
Epoch 3/5
50000/50000 [==============================] - 24s 486us/sample - loss: 1.5392 - accuracy: 0.4527
Epoch 4/5
50000/50000 [==============================] - 24s 485us/sample - loss: 1.4784 - accuracy: 0.4787
Epoch 5/5
50000/50000 [==============================] - 24s 485us/sample - loss: 1.4296 - accuracy: 0.4960
```

Baseline on a server with Nvidia A100 GPU:
```
Train on 50000 samples
2022-03-14 12:02:28.306455: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2022-03-14 12:02:29.499951: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1525] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 861 MB memory:  -> device: 0, name: NVIDIA A100-PCIE-40GB, pci bus id: 0000:3b:00.0, compute capability: 8.0
2022-03-14 12:02:29.500734: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1525] Created device /job:localhost/replica:0/task:0/device:GPU:1 with 1082 MB memory:  -> device: 1, name: NVIDIA A100-PCIE-40GB, pci bus id: 0000:af:00.0, compute capability: 8.0
Epoch 1/5
2022-03-14 12:02:30.896021: I tensorflow/stream_executor/cuda/cuda_blas.cc:1786] TensorFloat-32 will be used for the matrix multiplication. This will only be logged once.
50000/50000 [==============================] - 6s 125us/sample - loss: 1.8107 - accuracy: 0.3550
Epoch 2/5
50000/50000 [==============================] - 5s 97us/sample - loss: 1.6262 - accuracy: 0.4248
Epoch 3/5
50000/50000 [==============================] - 5s 98us/sample - loss: 1.5425 - accuracy: 0.4559
Epoch 4/5
50000/50000 [==============================] - 5s 96us/sample - loss: 1.4822 - accuracy: 0.4783
Epoch 5/5
50000/50000 [==============================] - 5s 98us/sample - loss: 1.4330 - accuracy: 0.4953
```

Baseline on the server using Intel i9 CPU:
```
Train on 50000 samples
2022-03-14 12:03:27.529990: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2022-03-14 12:03:28.747153: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1525] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 861 MB memory:  -> device: 0, name: NVIDIA A100-PCIE-40GB, pci bus id: 0000:3b:00.0, compute capability: 8.0
2022-03-14 12:03:28.747977: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1525] Created device /job:localhost/replica:0/task:0/device:GPU:1 with 1082 MB memory:  -> device: 1, name: NVIDIA A100-PCIE-40GB, pci bus id: 0000:af:00.0, compute capability: 8.0
Epoch 1/5
50000/50000 [==============================] - 34s 679us/sample - loss: 1.8070 - accuracy: 0.3575
Epoch 2/5
50000/50000 [==============================] - 34s 679us/sample - loss: 1.6191 - accuracy: 0.4296
Epoch 3/5
50000/50000 [==============================] - 34s 678us/sample - loss: 1.5401 - accuracy: 0.4551
Epoch 4/5
50000/50000 [==============================] - 34s 677us/sample - loss: 1.4805 - accuracy: 0.4778
Epoch 5/5
50000/50000 [==============================] - 34s 676us/sample - loss: 1.4335 - accuracy: 0.4943
```

So we can see that M1 GPU is one time faster than Intel Core i9 and is 25% faster than M1 CPU. But it can never compare with Nvidia XD.

