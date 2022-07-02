import itertools

import matplotlib
import matplotlib as mpl  # 画图用的库
import matplotlib.pyplot as plt
# 下面这一句是为了可以在notebook中画图
# %matplotlib inline
import numpy as np
import sklearn  # 机器学习算法库
import pandas as pd  # 处理数据的库
import os
import sys
import time
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow import keras  # 使用tensorflow中的keras
from sklearn import metrics

# import keras #单纯的使用keras
print(tf.__version__)
print(sys.version_info)
for module in mpl, np, sklearn, pd, tf, keras:
    print(module.__name__, module.__version__)

train_dir = "./input/training/training/"
valid_dir = "./input/validation/validation/"
label_file = "./input/monkey_labels.txt"
print(os.path.exists(train_dir))
print(os.path.exists(valid_dir))
print(os.path.exists(label_file))
print(os.listdir(train_dir))
print(os.listdir(valid_dir))
labels = pd.read_csv(label_file, header=0)
print(labels)

##################################
####该模块主要用于获取文件夹下的图片####
##################################
# resnet50使用的图像宽高均为224
# height = 224
# width = 224
height = 128  # 设置图像被缩放的宽高
width = 128
channels = 3  # 图像通道数
batch_size = 32
num_classes = 10
##########------------训练集数据------------##########
# 初始化一个训练数据相关的generator
# 具体用于 数据集中的图片数据进行处理，可以对图片数据进行归一化、旋转、翻转等数据增强类操作
train_datagen = keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=keras.applications.resnet50.preprocess_input,  # resnet50专门用来预处理图像的函数，把图像做归一化到-1~1之间
    # 使用第一行preprocessing_function 就不需要 rescale
    # rescale = 1./255, #放缩因子, 除以255是因为图片中每个像素点值范围都在0~255之间
    rotation_range=40,  # 图片随机转动的角度范围(-40 ~ 40)
    width_shift_range=0.2,  # 值 1时，表示像素宽度，即该图片的偏移幅度大小
    shear_range=0.2,  # 剪切强度
    zoom_range=0.2,  # 缩放强度
    horizontal_flip=True,  # 水平随机翻转
    fill_mode='nearest',  # 像素填充模式
)
# 接下来读取目录下的图片然后按照上面的数据增强相关操作对图片进行处理
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(height, width),  # 目录下的图片会被resize的大小
                                                    batch_size=batch_size,
                                                    seed=7,  # 随机种子，用于洗牌和转换，随便给个数即可
                                                    shuffle=True,  # False->则按字母数字顺序对数据进行排序 True->打乱数据
                                                    class_mode="categorical",  # 该参数决定了返回的标签数组的形式
                                                    # classes = 这个参数就是描述的 文件夹名与输出标签的对应关系
                                                    )
# classes：可选参数,为子文件夹的列表,如['dogs','cats']默认为None. 若未提供,则该类别列表将从directory下的子文件夹名称/结构自动推断。
# 每一个子文件夹都会被认为是一个新的类。(类别的顺序将按照字母表顺序映射到标签值)。通过属性class_indices可获得文件夹名与类的序号的对应字典。
# 使用生成器的.class_indices方法即可获取模型默认的Labels序列，文件夹名与类的序号的对应字典
print(train_generator.class_indices)
##########------------验证集数据------------##########
# 初始化一个验证数据相关的generator
# 验证数据集上不需要进行数据增强的相关操作，仅保留缩放即可，不然的话训练集与验证集的值的分布会不同
valid_datagen = keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=keras.applications.resnet50.preprocess_input,  # resnet50专门用来预处理图像的函数，相当于归一化，所以无需rescale
    # rescale = 1./255, #放缩因子, 除以255是因为图片中每个像素点值范围都在0~255之间
)
# 接下来读取目录下的图片然后按照上面的数据增强相关操作对图片进行处理
valid_generator = valid_datagen.flow_from_directory(valid_dir,
                                                    target_size=(height, width),  # 目录下的图片会被resize的大小
                                                    batch_size=batch_size,
                                                    seed=7,  # 随机种子，用于洗牌和转换，随便给个数即可
                                                    shuffle=False,  # 不需要训练所以不需要打乱数据
                                                    class_mode="categorical",  # 该参数决定了返回的标签数组的形式
                                                    )
# 使用生成器的.class_indices方法即可获取模型默认的Labels序列，文件夹名与类的序号的对应字典
print(valid_generator.class_indices)
train_num = train_generator.samples
valid_num = valid_generator.samples
print(train_num, valid_num)

# 使用resnet50做迁移学习

# 这里ResNet50中最后几层都是可以训练,我们可以在模型架构里面看到 Trainable params可训练参数会大大增加
resnet50 = keras.applications.ResNet50(include_top=False, pooling='avg', weights='imagenet')
for layers in resnet50.layers[0:-5]:  # 这里遍历最后五层之前的layers并设置其权重相关参数不可遍历
    layers.trainable = False
resnet50_fine_tune = keras.models.Sequential([
    resnet50,
    keras.layers.Dense(128, activation='relu'),  # 全连接层
    keras.layers.Dense(num_classes, activation='softmax'),  # 输出层
])
# 损失函数 sparse_categorical_crossentropy 和 categorical_crossentropy 的选择取决于前面设定的y值的取值类型
# 如果y取值为 2D的 one-hot编码，则选择 categorical_crossentropy
# 如果y取值为 1D的 整数标签，则选择 sparse_categorical_crossentropy
# 前面的 tensorflow2------分类问题fashion_mnist 文章中有过相关描述
# metrics 表示选择 accuracy作为评价参数
resnet50_fine_tune.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
resnet50_fine_tune.summary()

import shutil

callback_dir = "./callback_10-monkey-species"
if os.path.exists(callback_dir):
    shutil.rmtree(callback_dir)
    os.mkdir(callback_dir)

output_model_file = os.path.join(callback_dir, "10Monkey_model.h5")  # 在logdir中创建一个模型文件.h5
callbacks = [
    keras.callbacks.TensorBoard(callback_dir),
    keras.callbacks.ModelCheckpoint(output_model_file, save_best_only=True),
    keras.callbacks.EarlyStopping(patience=5, min_delta=1e-3),
]
epochs = 10  # 使用fine_tune 不需要太多次迭代就能够达到一个较好的效果
'''
#使用fit_generator是因为使用的是 ImageDataGenerator 获取数据集数据的
history = model.fit_generator(train_generator,#steps_per_epoch: 一个epoch包含的步数（每一步是一个batch的数据送入）
steps_per_epoch = train_num // batch_size,
epochs = epochs,
validation_data = valid_generator,
validation_steps= valid_num // batch_size,
callbacks = callbacks,
)
'''
history = resnet50_fine_tune.fit_generator(train_generator,  # steps_per_epoch: 一个epoch包含的步数（每一步是一个batch的数据送入）
                                           steps_per_epoch=train_num // batch_size,
                                           epochs=epochs,
                                           validation_data=valid_generator,
                                           validation_steps=valid_num // batch_size,
                                           callbacks=callbacks,
                                           )


# 运行打印看到val_accuracy的值并没有逐渐变大而是一直保持不变，是因为激活函数使用的是selu导致，可尝试更换激活函数为relu

def plot_learning_curves(history, label, epochs, min_value, max_value):
    data = {}
    data[label] = history.history[label]
    data['val_' + label] = history.history['val_' + label]
    pd.DataFrame(data).plot(figsize=(8, 5))
    plt.grid(True)
    plt.axis([0, epochs, min_value, max_value])
    plt.show()

#混淆矩阵
def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=None, normalize=False):
    accuracy = np.trace(cm) / float(np.sum(cm))  # 计算准确率
    misclass = 1 - accuracy  # 计算错误率
    if cmap is None:
        cmap = plt.get_cmap('Blues')  # 颜色设置成蓝色
    plt.figure(figsize=(10, 8))  # 设置窗口尺寸
    plt.imshow(cm, interpolation='nearest', cmap=cmap)  # 显示图片
    plt.title(title)  # 显示标题
    plt.colorbar()  # 绘制颜色条

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)  # x坐标标签旋转45度
        plt.yticks(tick_marks, target_names)  # y坐标

    if normalize:
        cm = cm.astype('float32') / cm.sum(axis=1)
        cm = np.round(cm, 2)  # 对数字保留两位小数

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]),
                                  range(cm.shape[1])):  # 将cm.shape[0]、cm.shape[1]中的元素组成元组，遍历元组中每一个数字
        if normalize:  # 标准化
            plt.text(j, i, "{:0.2f}".format(cm[i, j]),  # 保留两位小数
                     horizontalalignment="center",  # 数字在方框中间
                     color="white" if cm[i, j] > thresh else "black")  # 设置字体颜色
        else:  # 非标准化
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",  # 数字在方框中间
                     color="white" if cm[i, j] > thresh else "black")  # 设置字体颜色

    plt.tight_layout()  # 自动调整子图参数,使之填充整个图像区域
    plt.ylabel('True label')  # y方向上的标签
    plt.xlabel("Predicted label\naccuracy={:0.4f}\n misclass={:0.4f}".format(accuracy, misclass))  # x方向上的标签
    plt.show()  # 显示图片


plot_learning_curves(history, 'accuracy', epochs, 0, 1)
plot_learning_curves(history, 'loss', epochs, 0, 5)

monkey_names = ['mantled_howler',
                'patas_monkey',
                'bald_uakari',
                'japanese_macaque',
                'pygmy_marmoset',
                'white_headed_capuchin',
                'silvery_marmoset',
                'common_squirrel_monkey',
                'black_headed_night_monkey',
                'nilgiri_langur']

# 预测验证集数据整体准确率
Y_pred = resnet50_fine_tune.predict_generator(valid_generator, valid_num // batch_size + 1)
# 将预测的结果转化为one hit向量
Y_pred_classes = np.argmax(Y_pred, axis=1)
# 计算混淆矩阵
confusion_mtx = confusion_matrix(y_true=valid_generator.classes, y_pred=Y_pred_classes)
# 绘制混淆矩阵
plot_confusion_matrix(confusion_mtx, normalize=True, target_names=monkey_names)


