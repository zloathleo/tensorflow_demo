import tensorflow as tf
import tensorlayer as tl

#tensorlayer_demo1.py

# 第一步，建立会话 
sess = tf.InteractiveSession()

# 第二步，准备数据
X_train, y_train, X_val, y_val, X_test, y_test = \
                                tl.files.load_mnist_dataset(shape=(-1,784))

# 第三步，定义占位符，分别为网络的输入和目标输出 定义 placeholder
x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
y_ = tf.placeholder(tf.int64, shape=[None, ], name='y_')

# 第四步，建立模型 定义模型
network = tl.layers.InputLayer(x, name='input_layer')
# 对输入数据应用20%的退出率(dropout) 第一个参数是输入层，第二个参数是激活值的保持概率(keeping probability for the activation value) 
network = tl.layers.DropoutLayer(network, keep=0.8, name='drop1')
# 800个单位的全连接的隐藏层 n_units 简明得给出了全连接层的单位数 act 指定了一个激活函数
network = tl.layers.DenseLayer(network, n_units=800,
                                act = tf.nn.relu, name='relu1')
# 添加50%的退出率
network = tl.layers.DropoutLayer(network, keep=0.5, name='drop2')
# 800个单位的稠密层
network = tl.layers.DenseLayer(network, n_units=800,
                                act = tf.nn.relu, name='relu2')
# 添加50%的退出率
network = tl.layers.DropoutLayer(network, keep=0.5, name='drop3')

# 最后，我们加入 n_units 等于分类个数的全连接的输出层
network = tl.layers.DenseLayer(network, n_units=10,
                                act = tf.identity,
                                name='output_layer')

# 第五步，定义损失函数
# 定义损失函数和衡量指标
# tl.cost.cross_entropy 在内部使用 tf.nn.sparse_softmax_cross_entropy_with_logits() 实现 softmax
y = network.outputs
# 计算交叉熵损失，这个API调用tf的函数在内部实现了softmax
cost = tl.cost.cross_entropy(y, y_, name = 'cost')
# 判断预测结果和真实标签是否相同，得到的是一系列的布尔型Tensor
correct_prediction = tf.equal(tf.argmax(y, 1), y_)
# 计算准确率
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# y是one-hot形式，y_op得到的是代表类别的索引
y_op = tf.argmax(tf.nn.softmax(y), 1)

# 第六步，定义优化器
# 定义 optimizer
train_params = network.all_params
train_op = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999,
                            epsilon=1e-08, use_locking=False).minimize(cost, var_list=train_params)

# 第七步，开启Tensorboard
# 开启Tensorboard
acc_summ = tf.summary.scalar('acc', acc)
cost_summ = tf.summary.scalar('cost', cost)
summary = tf.summary.merge_all()
writer = tf.summary.FileWriter('./logs')
writer.add_graph(sess.graph)

# 第八步，初始化所有参数 
tl.layers.initialize_global_variables(sess)

# 第九步，列出模型信息
network.print_params()
network.print_layers()

# 第十步，训练模型 mini-batch大小为512，迭代100轮，每10轮打印一次训练信息
tl.utils.fit(sess, network, train_op, cost, X_train, y_train, x, y_,
            acc=acc, batch_size=512, n_epoch=100, print_freq=10,
            X_val=X_val, y_val=y_val, eval_train=False ,tensorboard=True)

# 第十一步，评估测试集
tl.utils.test(sess, network, acc, X_test, y_test, x, y_, batch_size=None, cost=cost)

# 第十二步，保存模型，关闭会话 把模型保存成 .npz 文件
tl.files.save_npz(network.all_params , name='model.npz')
sess.close()