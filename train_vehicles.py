import os
import argparse
import math

import tensorflow as tf
import numpy as np
# from tensorflow import keras
from tensorflow.keras.layers import Input
from sklearn.metrics import classification_report



AUTOTUNE = tf.data.experimental.AUTOTUNE
TRAINING_CHANNEL = "training"
VAL_CHANNEL = "validation"
TEST_CHANNEL = "test"
INPUT_SHAPE = 224
NUM_CLASSES = 5

# Features description for decoding tfrecords
features_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    

def efficientnet_model(input_shape, nb_classes):
    inputs = Input(shape=(input_shape, input_shape, 3))
    baseModel = tf.keras.applications.EfficientNetB1(include_top=False, weights='imagenet',
                                                    input_tensor=inputs)

    # Freeze the pretrained weights
    baseModel.trainable = False

    # Rebuild top
    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(baseModel.output)
    x = tf.keras.layers.BatchNormalization()(x)

    top_dropout_rate = 0.5
    x = tf.keras.layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = tf.keras.layers.Dense(nb_classes, activation="softmax", name="softmax")(x)

    # Compile
    model = tf.keras.Model(inputs, outputs, name="EfficientNet")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

#     model.summary()
    # input()
    return model

def decode_image(image, img_size=INPUT_SHAPE, channels=3):
    image = tf.image.decode_jpeg(image, channels=channels)
    image = tf.image.resize(image, [img_size, img_size])
    return image

def tfrecord_parser(record, num_classes=NUM_CLASSES):
    parsed = tf.io.parse_single_example(record, features_description)
    image = decode_image(parsed['image'])
    label = tf.one_hot(parsed['label'], num_classes, dtype=tf.float32)
    return image, label

def load_dataset(args, channel, shuffle=500):
    if channel == TRAINING_CHANNEL:
        data_dir = args.train_dir
        nb_images = 5818
    elif channel == VAL_CHANNEL:
        data_dir = args.val_dir
        nb_images = 1656
    
    elif channel == TEST_CHANNEL:
        data_dir = args.test_dir
        nb_images = 856
        
    print("datadir", data_dir)

    if args.pipe_mode == 1:
        ds = PipeModeDataset(channel=channel, record_format='TFRecord')

    else:
        tfrecords_files = tf.data.Dataset.list_files(os.path.join(data_dir, '*tfrecords'))
        print(len(tfrecords_files))
        ds = tf.data.TFRecordDataset(filenames=tfrecords_files, num_parallel_reads=AUTOTUNE)
    
    if not channel == TEST_CHANNEL:
        ds = (ds
            .map(tfrecord_parser, num_parallel_calls=AUTOTUNE)
            .shuffle(shuffle)
            .cache()
            .repeat()
            .batch(args.batch_size, drop_remainder=True)
            .prefetch(AUTOTUNE)

        )
    else:
        ds = (ds
            .map(tfrecord_parser, num_parallel_calls=AUTOTUNE)
            .cache()
            .batch(args.batch_size, drop_remainder=True)
            .prefetch(AUTOTUNE)
        )

    #TODO fix count for PipeMode
    # files_list = os.listdir(data_dir)
    # nb_images = 0
    # for filename in files_list:
    #     nb_images += int(filename.split('.')[0].split('-')[-1])
    # print("found %d for %s" %(nb_images, channel))
    # input()
    return ds, nb_images


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--pipe_mode", type=int, default=0)
    arg_parser.add_argument("--batch_size", type=int, default=64)
    arg_parser.add_argument("--epochs", type=int, default=2)

    arg_parser.add_argument("--train_dir", type=str)
    arg_parser.add_argument("--val_dir", type=str)
    arg_parser.add_argument("--test_dir", type=str)
    arg_parser.add_argument("--model_dir", type=str)
    args, _ = arg_parser.parse_known_args()

    if args.pipe_mode == 1:
        print("Pipe Mode")
        from sagemaker_tensorflow import PipeModeDataset
    else:
        print("File Mode")
    
    print("Loading Dataset")
    # Load training tfrecords
    train_ds, nb_train_imgs = load_dataset(args, TRAINING_CHANNEL)
    # Load validation tfrecords
    val_ds, nb_val_imgs = load_dataset(args, VAL_CHANNEL)
    #Load test tfrecords
    test_ds, nb_test_imgs = load_dataset(args, TEST_CHANNEL)
    test_labels = np.argmax(np.concatenate([y for x, y in test_ds], axis=0), axis = 1)

    # Compile Model
    model = efficientnet_model(INPUT_SHAPE, NUM_CLASSES)

    # Start Training
    print("Start training")
    history = model.fit(train_ds,
                    steps_per_epoch=math.ceil(nb_train_imgs/args.batch_size),
                    epochs=args.epochs,
                    validation_data=val_ds,
                    verbose = 2,
                    validation_steps=math.ceil(nb_val_imgs/args.batch_size)
                    )
    
    model.save(args.model_dir)
    
    #Testset Evaluation
    results = model.evaluate(test_ds, 
                        steps=math.ceil(nb_test_imgs/args.batch_size), 
                        verbose = 2,
                        batch_size=args.batch_size
                        )
    print(results)

    #Classification Report
    test_pred = model.predict(test_ds,
                        verbose = 2,
                        steps=math.ceil(nb_test_imgs/args.batch_size)
                        )
    target_names = ['bobtail', 'double_trailer', 'others', 'straight', 'trailer']
    print(classification_report(test_labels, 
                                np.argmax(test_pred, axis = 1),
                                target_names = target_names))