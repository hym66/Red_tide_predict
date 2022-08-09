from pickle import dump
import PIL
import numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
from tensorflow.keras.utils import  plot_model
from keras.models import load_model
import math
import sklearn.metrics as skm
from keras.layers import Flatten, BatchNormalization, Conv3D
from keras.losses import categorical_crossentropy, SparseCategoricalCrossentropy
from numpy import load
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers
import keras.backend as K
from keras.callbacks import LearningRateScheduler

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

import math
import sklearn.metrics as skm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.layers import RepeatVector, TimeDistributed
from tensorflow.keras.layers import ConvLSTM2D


def split_dataset(data):
    '''
    该函数实现以周为单位切分训练数据和测试数据
    '''
    len_train=int((len(data)*0.7))
    len_test=int((len(data)*0.3))

    train, test = data[0:len_train], data[-len_test:]
    # train = np.array(np.array_split(train, len(train) / 20))  # 将数据划分为按周为单位的数据
    # test = np.array(np.array_split(test, len(test) / 20))
    return train, test


def evaluate_forecasts(actual, predicted):
    '''
    该函数实现根据预期值评估一个或多个周预测损失
    思路：统计所有单日预测的 RMSE
    '''
    scores = list()
    for i in range(actual.shape[1]):
        mse = skm.mean_squared_error(actual[:, i], predicted[:, i])
        rmse = math.sqrt(mse)
        scores.append(rmse)

    s = 0  # 计算总的 RMSE
    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            s += (actual[row, col] - predicted[row, col]) ** 2
    score = math.sqrt(s / (actual.shape[0] * actual.shape[1]))
    print('actual.shape[0]:{}, actual.shape[1]:{}'.format(actual.shape[0], actual.shape[1]))
    return score, scores


def summarize_scores(name, score, scores):
    s_scores = ', '.join(['%.1f' % s for s in scores])
    print('%s: [%.3f] %s\n' % (name, score, s_scores))


def sliding_window(train, sw_width=7, n_out=7, in_start=0):
    '''
    该函数实现窗口宽度为7、滑动步长为1的滑动窗口截取序列数据
    '''

    data = train.reshape((train.shape[0] * train.shape[1],train.shape[2],train.shape[3]))  # 将以周为单位的样本展平为以天为单位的序列
    X, y = [], []

    for _ in range(len(data)):
        in_end = in_start + sw_width
        out_end = in_end + n_out

        # 保证截取样本完整，最大元素索引不超过原序列索引，则截取数据；否则丢弃该样本
        if out_end < len(data):
            # 训练数据以滑动步长1截取
            train_seq = data[in_start:in_end, :35,:]
            #train_seq = train_seq.reshape((len(train_seq)*35,13))
            X.append(train_seq)
            y.append(data[in_end:out_end, :35,0])
        in_start += 1

    return np.array(X), np.array(y)


def conv_lstm_model(train, sw_width, n_steps, n_length, in_start=0, verbose_set=0, epochs_num=20, batch_size_set=16):
    '''
    该函数定义 Encoder-Decoder LSTM 模型
    '''
    train_x, train_y = sliding_window(train, sw_width,7, in_start=0)
    #(13,490,35)  (13,1,35)
    n_timesteps, n_features, n_outputs = 7, 7, 35#changed
    #[样本，时间步长，行，列，通道]（[samples, timesteps, rows, cols, channels]）

    #4,7,35,13  4个周，
    train_x = train_x.reshape((train_x.shape[0], 2, 35, 7, n_features))
    train_y = train_y.reshape((train_y.shape[0], 1,35,7, 1))
    # model = Sequential()
    # model.add(ConvLSTM2D(filters=64, kernel_size=(13, 3), activation='relu',
    #                      input_shape=(2,35,7,n_features)))
    # model.add(Flatten())
    # print(model.output_shape)
    # model.add(RepeatVector(7))
    # model.add(LSTM(200, activation='relu', return_sequences=True))
    # model.add(TimeDistributed(Dense(100, activation='relu')))
    # model.add(TimeDistributed(Dense(1)))
    #
    # model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    seq = Sequential()
    seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                       input_shape=(None, 40, 40, 1),
                       padding='same', return_sequences=True))
    seq.add(BatchNormalization())

    seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                       padding='same', return_sequences=True))
    seq.add(BatchNormalization())

    seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                       padding='same', return_sequences=True))
    seq.add(BatchNormalization())

    seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                       padding='same', return_sequences=True))
    seq.add(BatchNormalization())

    seq.add(Conv3D(filters=1, kernel_size=(3, 3, 3),
                   activation='sigmoid',
                   padding='same', data_format='channels_last'))
    seq.compile(loss='binary_crossentropy', optimizer='adadelta')

    print(seq.summary())
    def scheduler(epoch):
        # 每隔100个epoch，学习率减小为原来的1/10
        if epoch % 10 == 0 and epoch!= 0:
            lr = K.get_value(seq.optimizer.lr)
            K.set_value(seq.optimizer.lr, lr * 0.1)
            print("lr changed to {}".format(lr * 0.1))
        return K.get_value(seq.optimizer.lr)

    reduce_lr = LearningRateScheduler(scheduler)
    seq.fit(train_x, train_y,
              epochs=epochs_num, batch_size=batch_size_set, verbose=verbose_set,callbacks=[reduce_lr])
    return seq


def forecast(model, pred_seq, sw_width, n_length, n_steps):
    '''
    该函数实现对输入数据的预测
    '''
    data = np.array(pred_seq)
    data = data.reshape((data.shape[0] * data.shape[1], data.shape[2],data.shape[3]))

    input_x = data[-sw_width:, :,:]  # 获取输入数据的最后一周的数据
    input_x = input_x.reshape((1, 2, 35, 7, 6))

    yhat = model.predict(input_x, verbose=0)  # 预测下周数据
    yhat = yhat[0]  # 获取预测向量
    return yhat


def evaluate_model(model, train, test, time_index,sd_width, n_length, n_steps):
    '''
    该函数实现模型评估
    '''
    history_fore = [x for x in train]
    predictions = list()  # 用于保存每周的前向验证结果；
    for i in range(len(test)):
        yhat_sequence = forecast(model, history_fore, sd_width, n_length, n_steps)  # 预测下周的数据
        predictions.append(yhat_sequence)  # 保存预测结果
        history_fore.append(test[i, :])  # 得到真实的观察结果并添加到历史中以预测下周
    predictions = np.array(predictions)  # 评估一周中每天的预测结果
    #把每个35*1生成一张图片
    for i in range(len(predictions)):
        for j in range(7):
            print(predictions[i][j])
            print(test[i][j])

            image_pre=PIL.Image.fromarray(predictions[i][j]).convert('P')
            img_pre = image_pre.resize((224, 224), PIL.Image.ANTIALIAS)
            img_pre.save("./image_pre/"+str(i)+"_"+str(j)+".png")
            image_actual = PIL.Image.fromarray(test[i][j]).convert('P')
            img_actual = image_actual.resize((224, 224), PIL.Image.ANTIALIAS)
            img_actual.save("./image_actual/" + str(i) + "_" + str(j) + ".png")



    #score, scores = evaluate_forecasts(test[:, 0, :,0], predictions)
    # return score, scores


def model_plot(score, scores, days, name):
    '''
    该函数实现绘制RMSE曲线图
    '''
    plt.figure(figsize=(8, 6), dpi=150)
    plt.plot(days, scores, marker='o', label=name)
    plt.grid(linestyle='--', alpha=0.5)
    plt.ylabel(r'$RMSE$', size=15)
    plt.title('Conv-LSTM 模型预测结果', size=18)
    plt.legend()
    plt.show()


def main_run(dataset, time_index,sw_width, days, name, in_start, verbose, epochs, batch_size, n_steps, n_length):
    '''
    主函数：数据处理、模型训练流程
    '''
    # 划分训练集和测试集
    train, test = split_dataset(dataset)
    # 训练模型
    model = conv_lstm_model(train, sw_width, n_steps, n_length, in_start, verbose_set=0, epochs_num=20,
                            batch_size_set=16)
    # 计算RMSE
    score, scores = evaluate_model(model, train, test, time_index,sw_width, n_length, n_steps)
    # 打印分数
    summarize_scores(name, score, scores)
    # 绘图
    model_plot(score, scores, days, name)
    print('绘图完成!')

#samples,num_frame=7,w,h,c
def create_shifted_frames(data):
    x = data[:, 0 : data.shape[1] - 1, :, :]
    y = data[:, 1 : data.shape[1], 0, :, :]
    return x, y

n_steps_in = 3
n_features = 7
n_steps_out = 1
def get_X_y(data):
    X = list()
    y = list()
    length = len(data)
    for i in range(0, length, 1):
        X_value = data[i: i + n_steps_in][:, :]
        y_value = data[i + n_steps_in: i + (n_steps_in + n_steps_out)][:, 0]
        if len(X_value) == 3 and len(y_value) == 1:
            X.append(X_value)
            y.append(y_value)
    return np.array(X), np.array(y)
if __name__ == '__main__':
#    -------------------分割线--------------------------
    dataset=numpy.load("time_list1.npy")

    #按照7：3划分成按周的数据集  samples,num_frame=7,w,h,c
    X,y=get_X_y(dataset)
    # train,test=split_dataset(dataset)
    train_x,test_x=split_dataset(X)
    train_y,test_y=split_dataset(y)
    # Inspect the dataset.
    print("Training Dataset Shapes: " + str(train_x.shape) + ", " + str(train_y.shape))
    print("Validation Dataset Shapes: " + str(test_x.shape) + ", " + str(test_y.shape))
    # 定义序列的数量和长度

        # Construct a figure on which we will visualize the images.
    fig, axes = plt.subplots(1, 3, figsize=(7, 10))#7,10

    # Plot each of the sequential images for one random data example.d
    data_choice = np.random.choice(range(len(train_x)), size=1)[0]
    for idx, ax in enumerate(axes.flat):
        train_pic=np.squeeze(train_x[data_choice][idx][0][:][:])
        ax.imshow(train_pic, cmap="gray")
        ax.set_title(f"Frame {idx + 1}")
        ax.axis("off")

    # Print information and display the figure.
    print(f"Displaying frames for example {data_choice}.")
    plt.show()
    train_x=train_x.reshape(train_x.shape[0],train_x.shape[1], 40, 20, 6)#变量
    train_y = train_y.reshape(train_y.shape[0], train_y.shape[1], 40, 20, 1)#chl
    test_x = test_x.reshape(test_x.shape[0], test_x.shape[1], 40, 20, 6)
    test_y = test_y.reshape(test_y.shape[0], test_y.shape[1], 40, 20, 1)

    #开始建模了就是说
    # Construct the input layer with no definite frame size.
    #train_x  (sample,6,13,7,5)
    inp = layers.Input(shape=(None, *train_x.shape[2:]))

    # We will construct 3 `ConvLSTM2D` layers with batch normalization,
    # followed by a `Conv3D` layer for the spatiotemporal outputs.
    x = layers.ConvLSTM2D(
        filters=32,
        kernel_size=(5, 5),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(inp)
    #none none 11 5 5
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(
        filters=32,
        kernel_size=(5,5),
        padding="same",
        return_sequences=True,
        activation="sigmoid",
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(
        filters=32,
        kernel_size=(5, 5),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(x)
    x = layers.Conv3D(
        filters=1, kernel_size=(3, 3, 3),activation="relu", padding="same"
    )(x)

    # Next, we will build the complete model and compile it.
    model = keras.models.Model(inp, x)
    model.compile(
        loss='mean_squared_error',metrics=['accuracy'], optimizer=keras.optimizers.Adam(lr=0.0001),
    )

    # Define some callbacks to improve training.
    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)

    # Define modifiable training hyperparameters.
    epochs = 10
    batch_size = 16

    # Fit the model to the training data.
    model.fit(
        train_x,
        train_y,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(test_x, test_y),
        callbacks=[early_stopping, reduce_lr],
    )
plot_model(model,to_file="model.png",show_shapes=True)

# Select a random example from the validation dataset.
example = test_x
frames=[]
#test假如有500天，那就需要预测500次欸 不断把预测结果加到答案里 得到一个集合
for day in range(len(test_x)):
    new_prediction=model.predict(test_x[day].reshape(1,3,40,20,6))
    frames.append(new_prediction)
frames=np.array(frames)

# Predict a new set of 10 frames.
# for _ in range(10):
#     # Extract the model's prediction and post-process it.
#     new_prediction = model.predict(np.expand_dims(frames, axis=0))
#     new_prediction = np.squeeze(new_prediction, axis=0)
#     predicted_frame = np.expand_dims(new_prediction[-1, ...], axis=0)
#
#     # Extend the set of prediction frames.
#     frames = np.concatenate((frames, predicted_frame), axis=0)
####################################################################################################################################################
temp_f=frames.reshape(frames.shape[0]*frames.shape[1]*frames.shape[2]*frames.shape[3]*frames.shape[4],frames.shape[5])
temp_test=test_y.reshape(test_y.shape[0]*test_y.shape[1]*test_y.shape[2]*test_y.shape[3],test_y.shape[4])


scalar=load(open('scaler_y.pkl', 'rb'),allow_pickle=True)
tf=scalar.inverse_transform(temp_f)
tt=scalar.inverse_transform(temp_test)
temp_f=tf.reshape(frames.shape[0],frames.shape[1],frames.shape[2],frames.shape[3],frames.shape[4])
temp_test=tt.reshape(test_y.shape[0],test_y.shape[1],test_y.shape[2],test_y.shape[3],test_y.shape[4])
# Construct a figure for the original and new frames.
fig, axes = plt.subplots(2,3, figsize=(20, 4))
option=np.random.choice(len(temp_f))
pre=np.squeeze(frames[option,...])
actual=np.squeeze(test_x[option,...,0])

#把高维的数据展平为二维

# Plot the original frames.
for idx, ax in enumerate(axes[0]):
    ax.imshow(pre[idx], cmap="gray")
    ax.set_title(f"Frame {idx + 11}")
    ax.axis("off")

# Plot the new frames.
# new_frames = frames[10:, ...]
for idx, ax in enumerate(axes[1]):
    ax.imshow(actual[idx], cmap="gray")
    ax.set_title(f"Frame {idx + 11}")
    ax.axis("off")

# Display the figure.
plt.show()

# --------------------分割线------------------------
#$example是验证集中抽出来的一组图片
#example = val_dataset[np.random.choice(range(len(val_dataset)), size=1)[0]]
#example=test[:][:]
# Pick the first/last 2 frames from the example.
# frames = example[:2, ...]
# original_frames = example[:, ...]
#
# # Predict a new set of 10 frames.
# for _ in range(2):
#     # Extract the model's prediction and post-process it.
#     pre=np.expand_dims(frames, axis=0).squeeze()
#
#     new_prediction = model.predict(pre)
#     frames = np.concatenate((frames, new_prediction), axis=0)
#
# original_frames=original_frames
# # Construct a figure for the original and new frames.
# fig, axes = plt.subplots(2, 7, figsize=(20, 4))
# i=0
# # Plot the original frames.
# for idx, ax in enumerate(axes[0]):
#     ax.imshow(np.squeeze(original_frames[0,idx,:,:,0]), cmap="gray")
#     ax.set_title(f"Frame {idx + 11}")
#     ax.axis("off")
#     i+=1
#
# i=0
#
# # Plot the new frames.
# new_frames = frames[3:, ...]
# for idx, ax in enumerate(axes[1]):
#     ax.imshow(np.squeeze(new_frames[0,idx,:,:,0]), cmap="gray")
#     ax.set_title(f"Frame {idx + 11}")
#     ax.axis("off")
# i=0
# # Display the figure.
# plt.show()



