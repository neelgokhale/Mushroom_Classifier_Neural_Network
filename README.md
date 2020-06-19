# Mushroom_Classifier_Neural_Network

This neural network takes 23 mushroom features and characteristics as inputs and predicts if the mushroom is poisonous or safely edible. Please DO NOT abide by this program in actual mushroom sampling...

## Imports and Set-up


```python
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers
import pandas as pd
from IPython.display import clear_output
import matplotlib.pyplot as plt
import os
import numpy as np
from tensorflow import estimator
```

## Data Import and Formatting


```python
CSV_COLUMN_NAMES = ['result', 'cap-shape', 'cap-surf', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-num', 'ring-type', 'spore-print-color', 'population', 'habitat']

train_path = os.path.join("data", "agaricus-lepiota.csv")
train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0, skiprows=1)
test = train[7000:]
train = train[0:7000]
train.head(100)

train_y, test_y = train.pop('result'), test.pop('result')

train
train_y_num = []
test_y_num = []

for i in train_y:
    if i == 'e':
        train_y_num.append(1)
    else:
        train_y_num.append(0)

for i in test_y:
    if i == 'e':
        test_y_num.append(1)
    else:
        test_y_num.append(0)
```

## Input Function


```python
def input_fn(features, labels, training=True, batch_size=1):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    
    if training:
        dataset = dataset.shuffle(1000).repeat()
    return dataset.batch(batch_size)

```


```python
def demo(feature_column):
  feature_layer = layers.DenseFeatures(feature_column)
  print(feature_layer(train.to_dict('series')).numpy())
```

## Defining Feature Columns Using One Hot Columns


```python
COLUMN_NAMES = {'cap-shape':['b', 'c', 'f', 'x', 'k', 's'],
                'cap-surf':['f', 'g', 'y', 's'],
                'cap-color':['n', 'b', 'c', 'g', 'r', 'p', 'u', 'e', 'w', 'y'],
                'bruises':['t', 'f'],
                'odor':['a', 'l', 'c', 'y', 'f', 'm', 'n', 'p', 's'],
                'gill-attachment':['a', 'd', 'f', 'n'],
                'gill-spacing':['c', 'w', 'd'],
                'gill-size':['b', 'n'],
                'gill-color':['k', 'n', 'b', 'h', 'g', 'r', 'o', 'p', 'u', 'e', 'w', 'y'],
                'stalk-shape':['e', 't'],
                'stalk-root':['b', 'c', 'u', 'e', 'z', 'r', '?'],
                'stalk-surface-above-ring':['f', 'y', 'k', 's'],
                'stalk-surface-below-ring':['f', 'y', 'k', 's'],
                'stalk-color-above-ring':['n', 'b', 'c', 'g', 'o', 'p', 'u', 'e', 'w', 'y'],
                'stalk-color-below-ring':['n', 'b', 'c', 'g', 'o', 'p', 'u', 'e', 'w', 'y'],
                'veil-type':['p', 'u'], 
                'veil-color':['n', 'o', 'w', 'y'],
                'ring-num':['n', 'o', 't'],
                'ring-type':['c', 'e', 'f', 'l', 'n', 'p', 's', 'z'], 
                'spore-print-color':['k', 'n', 'b', 'h', 'r', 'o', 'u', 'w', 'y'], 
                'population':['k', 'n', 'b', 'h', 'r', 'o', 'u', 'w', 'y'], 
                'habitat':['g', 'l', 'm', 'p', 'u', 'w', 'd']
               }

feature_columns = []

for feature_name in COLUMN_NAMES:
    feature_columns.append(feature_column.indicator_column(feature_column.categorical_column_with_vocabulary_list(feature_name, COLUMN_NAMES[feature_name])))

feature_columns


```

## Classifier and Training


```python
classifier = tf.estimator.DNNClassifier(

    feature_columns=feature_columns,
    hidden_units=[100, 30],
    n_classes=2
)
```

    INFO:tensorflow:Using default config.
    WARNING:tensorflow:Using temporary folder as model directory: C:\Users\Owner\AppData\Local\Temp\tmp434zw6k3
    INFO:tensorflow:Using config: {'_model_dir': 'C:\\Users\\Owner\\AppData\\Local\\Temp\\tmp434zw6k3', '_tf_random_seed': None, '_save_summary_steps': 100, '_save_checkpoints_steps': None, '_save_checkpoints_secs': 600, '_session_config': allow_soft_placement: true
    graph_options {
      rewrite_options {
        meta_optimizer_iterations: ONE
      }
    }
    , '_keep_checkpoint_max': 5, '_keep_checkpoint_every_n_hours': 10000, '_log_step_count_steps': 100, '_train_distribute': None, '_device_fn': None, '_protocol': None, '_eval_distribute': None, '_experimental_distribute': None, '_experimental_max_worker_delay_secs': None, '_session_creation_timeout_secs': 7200, '_service': None, '_cluster_spec': ClusterSpec({}), '_task_type': 'worker', '_task_id': 0, '_global_id_in_cluster': 0, '_master': '', '_evaluation_master': '', '_is_chief': True, '_num_ps_replicas': 0, '_num_worker_replicas': 1}
    


```python
classifier.train(
    input_fn=lambda: input_fn(train, train_y_num, training=True),
    steps=7000
)
```

## Evaluation


```python
eval_result = classifier.evaluate(input_fn=lambda: input_fn(test, test_y_num, training=False))

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
```

    INFO:tensorflow:Calling model_fn.
    INFO:tensorflow:Done calling model_fn.
    INFO:tensorflow:Starting evaluation at 2020-06-19T10:10:13Z
    INFO:tensorflow:Graph was finalized.
    INFO:tensorflow:Restoring parameters from C:\Users\Owner\AppData\Local\Temp\tmp434zw6k3\model.ckpt-7000
    INFO:tensorflow:Running local_init_op.
    INFO:tensorflow:Done running local_init_op.
    INFO:tensorflow:Inference Time : 2.23038s
    INFO:tensorflow:Finished evaluation at 2020-06-19-10:10:15
    INFO:tensorflow:Saving dict for global step 7000: accuracy = 0.7139037, accuracy_baseline = 0.5864528, auc = 0.9921801, auc_precision_recall = 0.9875987, average_loss = 0.40659487, global_step = 7000, label/mean = 0.41354725, loss = 0.40659487, precision = 1.0, prediction/mean = 0.21231274, recall = 0.30818966
    INFO:tensorflow:Saving 'checkpoint_path' summary for global step 7000: C:\Users\Owner\AppData\Local\Temp\tmp434zw6k3\model.ckpt-7000
    
    Test set accuracy: 0.714
    
    

## Prediction


```python
def input_fn(features, batch_size=256):
    # Convert inputs to a dataset without labels
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

predict = {}
result = ['poisonous', 'edible']
print("Please type features")
for feature in CSV_COLUMN_NAMES[1:]:
    
    val = input(feature + ": ")
    
    predict[feature] = [val]

predictions = classifier.predict(input_fn=lambda: input_fn(predict))

for pred_dict in predictions:
    print(pred_dict)
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]
    
    print('The mushroom is "{}" with {:.1f}% confidence'.format(result[class_id], 100 * probability))
    
```

    Please type features
    

    cap-shape:  x
    cap-surf:  y
    cap-color:  y
    bruises:  t
    odor:  a
    gill-attachment:  f
    gill-spacing:  c
    gill-size:  b
    gill-color:  g
    stalk-shape:  e
    stalk-root:  c
    stalk-surface-above-ring:  s
    stalk-surface-below-ring:  s
    stalk-color-above-ring:  w
    stalk-color-below-ring:  w
    veil-type:  p
    veil-color:  w
    ring-num:  o
    ring-type:  pk
    spore-print-color:  s
    population:  m
    habitat:  s
    

    INFO:tensorflow:Calling model_fn.
    INFO:tensorflow:Done calling model_fn.
    INFO:tensorflow:Graph was finalized.
    INFO:tensorflow:Restoring parameters from C:\Users\Owner\AppData\Local\Temp\tmp434zw6k3\model.ckpt-7000
    INFO:tensorflow:Running local_init_op.
    INFO:tensorflow:Done running local_init_op.
    {'logits': array([0.17652605], dtype=float32), 'logistic': array([0.54401726], dtype=float32), 'probabilities': array([0.45598274, 0.54401726], dtype=float32), 'class_ids': array([1], dtype=int64), 'classes': array([b'1'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}
    The mushroom is "edible" with 54.4% confidence
    
