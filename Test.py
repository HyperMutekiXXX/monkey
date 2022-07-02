import matplotlib
import matplotlib as mpl  # 画图用的库
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
# from tensorflow.keras.models import load_model
def preprocess_img(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [128, 128])
    image /= 255.0
    return image


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_img(image)


image_path_test = './test/mm.jpg'
image1 = load_and_preprocess_image(image_path_test)


from tensorflow.keras.preprocessing import image
import pprint

height = 128
width = 128
img = image.load_img(image_path_test, target_size=(height, width))
img = image.img_to_array(img)
print(img.shape)  # 这里直接打印将img转换为数组后的数据维度 (128,128,3)
# 因为模型的输入是要求四维的，所以我们需要将输入图片增加一个维度，使用 expand_dims接口
img = np.expand_dims(img, axis=0)
print(img.shape)
new_model = tf.keras.models.load_model('./callback_10-monkey-species/10Monkey_model.h5')
# predict表示预测输出当前输入图像的 所有类型概率数组，即包含十个概率值的数组
pred = new_model.predict(img)
# pprint.pprint(pred)
print(pred)
print(np.argmax(pred, axis=1))  # axis = 1是取行的最大值的索引，axis = 0是列的最大值的索引
# predict_classes 预测的是类别，打印出来的值就是类别号
pred_class = new_model.predict_classes(img)
print(pred_class)
# 建立对应的文件夹排序的标签数组打印出预测的标签
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
label_name = [monkey_names[index] for index in pred_class]
# print("这是", ''.join(label_name))  # list转换为string
plt.imshow(image1)
plt.xlabel(label_name)
plt.show()