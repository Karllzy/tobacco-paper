import numpy as np
import scipy.io
import datetime


class ELM:

    """
    假设输入的数据 x 为一个 n个特征的数据 (1, input_size)

    ELM 的结构包括三部分
        1.一个隐含层权重 w （input_size, node_num）
        2.该隐含层的偏置值 b （1, node_num）
        3.后续加权值 beta (node_num, output_num)

    则进行预测时, h = w_(n, node_num).T . b
    """

    def __init__(self, input_size=10, node_num=14, output_num=4,
                 weight=None, bias=None, beta=None, rand_seed=None):
        """
        Initialize a ELM with 3 parameters: input_size, node_num, output_num
        the structure of an ELM is very simple:

        X_(nxN)  W_(NxL)+bias(1xL)->f=H(nxL).beta(Lxt)=T(nxt)
        x_data  ---------> Neurons --------------> Output

        :param input_size: [int] the feature numbers of input data e.g. [x1, x2, x3] ---> 3
        :param node_num: [int] the number of neurons in the hidden layer
        :param output_num: [int] the number of output
        :param weight: [array] shape: input_size x node_num
        :param bias: [array] shape: 1 x node_num
        :param beta: [array] shape: node_num, output_num
        :param rand_seed: [int] the random seed
        """
        if rand_seed is not None:
            np.random.seed(rand_seed)

        if weight is not None:
            self.w = weight
        else:
            # 随机生产权重（从-1，到1，生成（num_feature行,num_hidden列））
            self.w = np.random.uniform(0, 1, (input_size, node_num))

        if bias is not None:
            self.b = bias
        else:
            self.b = np.random.uniform(0, 1, (1, node_num))

        if beta is not None:
            self.beta = beta
        else:
            self.beta = np.random.uniform(0, 1, (node_num, output_num))

    def sigmoid(self, x):
        """
        sigmoid activation function
        :param x: input x
        :return: sigmoid output
        """
        return 1.0 / (1 + np.exp(-x))

    def fit(self, x_train, y_train, c=None):
        """
        fit the data
        :param x_train: [array] train data x
        :param y_train: [array] train data y
        :param c: regular c
        :return: self
        """
        mul = np.dot(x_train, self.w)  # 输入乘以权重
        add = mul + self.b             # 加偏置
        H = self.sigmoid(add)          # 激活函数
        HH = H.T.dot(H)
        HT = H.T.dot(y_train)
        node_num = self.w.shape[1]
        if c is None:
            self.beta = np.linalg.pinv(HH).dot(HT)
        else:
            self.beta = np.linalg.pinv(HH + np.identity(node_num) / c).dot(HT)
        return self

    def predict(self, x_data):
        """
        make prediction
        :param x_data: data to predict
        :return: predicted data
        """
        mul = np.dot(x_data, self.w)  # 输入乘以权重
        add = mul + self.b  # 加偏置
        H = self.sigmoid(add)  # 激活函数
        result = H.dot(self.beta)
        return result

    def save(self, path=None):
        """
        save the model
        :param path: model path
        :return: saved model name
        """
        if path is None:
            path = datetime.datetime.now().strftime("ELM_%Y-%m-%d_%H-%M.mat")
        model_dic = {"w": self.w, "b": self.b, "beta": self.beta}
        print(self.w.shape, self.b.shape, self.beta.shape)
        scipy.io.savemat(path, model_dic)
        return path

    def load(self, model_path):
        """
        load from saved model ([.mat] contain w, b, beta)
        :param model_path: load from where
        :return: self
        """
        data = scipy.io.loadmat(model_path)
        self.w, self.b, self.beta = data['w'], data['b'], data['beta'].T
        print(self.w.shape, self.b.shape, self.beta.shape)
        return self


if __name__ == "__main__":
    x1 = np.linspace(1, 20, 100)
    x2 = np.linspace(-5, 5, 100)
    X = np.vstack([x1, x2]).T
    T = np.sin(x1 * x2 / (2 * np.pi)) + np.random.normal(0, 0.2, 100)

    elm = ELM(input_size=2, node_num=100, output_num=1)

    elm.fit(X, T)
    # elm.load('ELM_2020-01-10_00-18.mat')
    y = elm.predict(X)
    # elm.save()

    import matplotlib.pyplot as plt
    plt.plot(x1, T, lw=1.5, label='Training goal')
    plt.plot(x1, y, lw=3, label='ELM output')
    plt.legend()
    plt.show()




