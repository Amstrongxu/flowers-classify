# _*_ coding:utf-8 _*_
# @Time : 2021/11/3 12:11
# @Author : xupeng
# @File : dataset_preprocess.py
# @software : PyCharm

#本例中使用的数据集分布在图片文件夹中，一个文件夹含有一类图片。
import tensorflow as tf
AUTOTUNE = tf.data.experimental.AUTOTUNE
import pathlib
import os
import matplotlib.pyplot as plt
from PIL import Image

import random

# 载并检查数据集
data_root_orig = tf.keras.utils.get_file(origin='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
                                         fname='flower_photos', untar=True)
# print(data_root_orig)
data_root = pathlib.Path(data_root_orig)
# print(data_root)  #C:\Users\28954\.keras\datasets\flower_photos

# for item in data_root.iterdir():
#     print(item)

all_image_paths = list(data_root.glob('*/*'))
# print(all_image_paths)
all_image_paths = [str(path) for path in all_image_paths]
random.shuffle(all_image_paths)
# print(all_image_paths)
image_count = len(all_image_paths)
# print(image_count)

# 检查图片
attributions = open(os.path.join(data_root,"LICENSE.txt"), encoding='utf-8').readlines()[4:]
# print(attributions)
attributions = [line.split(' CC-BY') for line in attributions]
# print(attributions)
attributions = dict(attributions)
# print(attributions)
#字典的一个元素： 'tulips/2481827798_6087d71134.jpg': ' by Christine Majul - https://www.flickr.com/photos/kitkaphotogirl/2481827798/\n',


def caption_image(image_path):
    image_rel = pathlib.Path(image_path).relative_to(data_root)
    return "Image (CC BY 2.0) " + ' - '.join(attributions[str(image_rel).replace('\\', '/')].split(' - ')[:-1])

for n in range(3):
    image_path = random.choice(all_image_paths)
    img = Image.open(image_path)
    plt.imshow(img)
    plt.show()

    print(caption_image(image_path))
    print()

# 确定每张图片的标签
label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
# print(label_names) #['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
# 为每个标签分配索引：
label_to_index = dict((name, index) for index, name in enumerate(label_names))
# print(label_to_index)
#{'daisy': 0, 'dandelion': 1, 'roses': 2, 'sunflowers': 3, 'tulips': 4}

# 创建一个列表，包含每个文件的标签索引：
all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                    for path in all_image_paths]

print("First 10 labels indices: ", all_image_labels[:10])

# 加载和格式化图片
img_path = all_image_paths[0]
print(img_path)
# 以下是原始数据：
img_raw = tf.io.read_file(img_path)
print(repr(img_raw)[:100] + "...")

# 将它解码为图像 tensor（张量）：
img_tensor = tf.image.decode_image(img_raw)
print(img_tensor.shape)
print(img_tensor.dtype)

# 根据你的模型调整其大小：
img_final = tf.image.resize(img_tensor, [192, 192])
img_final = img_final/255.0
print(img_final.shape)
print(img_final.numpy().min())
print(img_final.numpy().max())

# 将这些包装在一个简单的函数里，以备后用。
def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [192, 192])
    image /= 255.0  # normalize to [0,1] range

    return image

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)

image_path = all_image_paths[0]
label = all_image_labels[0]

plt.imshow(load_and_preprocess_image(img_path))
plt.grid(False)
plt.xlabel(caption_image(img_path))
plt.title(label_names[label].title())
plt.show()
print()

# 构建一个 tf.data.Dataset
# 构建 tf.data.Dataset 最简单的方法就是使用 from_tensor_slices 方法。
path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
print(path_ds)

# 现在创建一个新的数据集，通过在路径数据集上映射 preprocess_image 来动态加载和格式化图片。
image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

plt.figure(figsize=(8,8))
for n, image in enumerate(image_ds.take(4)):
    plt.subplot(2,2,n+1)
    plt.imshow(image)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(caption_image(all_image_paths[n]))
    plt.show()


# 一个(图片, 标签)对数据集
# 使用同样的 from_tensor_slices 方法你可以创建一个标签数据集：
label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))
for label in label_ds.take(10):
    print(label_names[label.numpy()])

# 由于这些数据集顺序相同，你可以将他们打包在一起得到一个(图片, 标签)对数据集
image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
print(image_label_ds) #<ZipDataset shapes: ((192, 192, 3), ()), types: (tf.float32, tf.int64)>

ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))
# 元组被解压缩到映射函数的位置参数中
def load_and_preprocess_from_path_label(path, label):
    return load_and_preprocess_image(path), label

image_label_ds = ds.map(load_and_preprocess_from_path_label)
print(image_label_ds)

# 训练的基本方法
BATCH_SIZE = 32
# 设置一个和数据集大小一致的 shuffle buffer size（随机缓冲区大小）以保证数据
# 被充分打乱。
ds = image_label_ds.shuffle(buffer_size=image_count)
ds = ds.repeat()
ds = ds.batch(BATCH_SIZE)
# 当模型在训练的时候，`prefetch` 使数据集在后台取得 batch。
ds = ds.prefetch(buffer_size=AUTOTUNE)
print(ds)

# 在随机缓冲区完全为空之前，被打乱的数据集不会报告数据集的结尾。
# Dataset（数据集）由 .repeat 重新启动，导致需要再次等待随机缓冲区被填满。
# 最后一点可以通过使用 tf.data.Dataset.apply 方法和融合过的
# tf.data.experimental.shuffle_and_repeat 函数来解决:
ds = image_label_ds.apply(
  tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
ds = ds.batch(BATCH_SIZE)
ds = ds.prefetch(buffer_size=AUTOTUNE)
print(ds)

# 传递数据集至模型
# 从 tf.keras.applications 取得 MobileNet v2 副本。
#
# 该模型副本会被用于一个简单的迁移学习例子。
#
# 设置 MobileNet 的权重为不可训练：
mobile_net = tf.keras.applications.MobileNetV2(input_shape=(192, 192, 3), include_top=False)
mobile_net.trainable=False

# 在你将输出传递给 MobilNet 模型之前，你需要将其范围从 [0,1] 转化为 [-1,1]：
def change_range(image,label):
  return 2*image-1, label

keras_ds = ds.map(change_range)

# MobileNet 为每张图片的特征返回一个 6x6 的空间网格。
image_batch, label_batch = next(iter(keras_ds))
feature_map_batch = mobile_net(image_batch)
print(feature_map_batch.shape)

# 构建一个包装了 MobileNet 的模型
model = tf.keras.Sequential([
  mobile_net,
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(len(label_names), activation = 'softmax')])

logit_batch = model(image_batch).numpy()

print("min logit:", logit_batch.min())
print("max logit:", logit_batch.max())
print()

print("Shape:", logit_batch.shape)

# 编译模型以描述训练过程：
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=["accuracy"])
# 此处有两个可训练的变量 —— Dense 层中的 weights（权重） 和 bias（偏差）：
print(len(model.trainable_variables))
model.summary()

# 训练模型
# 一般来说在传递给 model.fit() 之前你会指定 step 的真实数量
steps_per_epoch=tf.math.ceil(len(all_image_paths)/BATCH_SIZE).numpy()
print(steps_per_epoch)

# 出于演示目的每一个 epoch 中你将只运行 3 step
# model.fit(ds, epochs=1, steps_per_epoch=3)
model.fit(ds, epochs=5, steps_per_epoch=steps_per_epoch)

# 性能
# 上面使用的简单 pipeline（管道）在每个 epoch 中单独读取每个文件。
# 在本地使用 CPU 训练时这个方法是可行的，
# 但是可能不足以进行 GPU 训练并且完全不适合任何形式的分布式训练。
import time
default_timeit_steps = 2*steps_per_epoch+1

def timeit(ds, steps=default_timeit_steps):
    overall_start = time.time()
    # 在开始计时之前
    # 取得单个 batch 来填充 pipeline（管道）（填充随机缓冲区）
    it = iter(ds.take(steps+1))
    next(it)

    start = time.time()
    for i,(images,labels) in enumerate(it):
        if i%10 == 0:
            print('.',end='')
    print()
    end = time.time()

    duration = end-start
    print("{} batches: {} s".format(steps, duration))
    print("{:0.5f} Images/s".format(BATCH_SIZE*steps/duration))
    print("Total time: {}s".format(end-overall_start))

# 当前数据集的性能是：
ds = image_label_ds.apply(
  tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
ds = ds.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
print(timeit(ds))


# 缓存
# 使用 tf.data.Dataset.cache 在 epoch 之间轻松缓存计算结果。这是非常高效的，特别是当内存能容纳全部数据时。
# 使用内存缓存的一个缺点是必须在每次运行时重建缓存，这使得每次启动数据集时有相同的启动延迟：
ds = image_label_ds.cache()
ds = ds.apply(
  tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
ds = ds.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
print(timeit(ds))


# 如果内存不够容纳数据，使用一个缓存文件：
ds = image_label_ds.cache(filename='./cache.tf-data')
ds = ds.apply(
  tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
ds = ds.batch(BATCH_SIZE).prefetch(1)
print(timeit(ds))

# TFRecord 文件
# TFRecord 文件是一种用来存储一串二进制 blob 的简单格式。通过将多个示例打包进同一个文件内，
# TensorFlow 能够一次性读取多个示例，当使用一个远程存储服务，如 GCS 时，
# 这对性能来说尤其重要
# 首先，从原始图片数据中构建出一个 TFRecord 文件：
image_ds = tf.data.Dataset.from_tensor_slices(all_image_paths).map(tf.io.read_file)
tfrec = tf.data.experimental.TFRecordWriter('images.tfrec')
tfrec.write(image_ds)

# 接着，构建一个从 TFRecord 文件读取的数据集，
# 并使用你之前定义的 preprocess_image 函数对图像进行解码/重新格式化：
image_ds = tf.data.TFRecordDataset('images.tfrec').map(preprocess_image)

# 压缩该数据集和你之前定义的标签数据集以得到期望的 (图片,标签) 对：
ds = tf.data.Dataset.zip((image_ds, label_ds))
ds = ds.apply(
  tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
ds=ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
print(ds)
print(timeit(ds))