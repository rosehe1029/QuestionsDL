# QuestionsDL
百面深度学习所有问题带答案整理
  
《百面深度学习》所有问题带答案整理 
  
参考资料：
  第1章 卷积神经网络
1.简述卷积的基本操作，并分析其与全连接层的区别
https://zhuanlan.zhihu.com/p/474159361
https://blog.csdn.net/m0_51607165/article/details/123965085
2.在卷积神经网络中，如何计算各层的感受野大小?卷积层的输出尺寸、参数量和计算量。
https://zhuanlan.zhihu.com/p/44106492
https://blog.csdn.net/qqliuzihan/article/details/78079758
https://blog.csdn.net/mzpmzk/article/details/86564509
https://zhuanlan.zhihu.com/p/395354063
3.简述分组卷积及其应用场景
https://zhuanlan.zhihu.com/p/474159361
4.简述空洞卷积的设计思路
https://zhuanlan.zhihu.com/p/113285797
https://blog.csdn.net/YOULANSHENGMENG/article/details/121208470
5.简述转置卷积的主要思想以及应用场景
https://zhuanlan.zhihu.com/p/115070523
6.可变形卷积旨在解决哪类问题?
https://zhuanlan.zhihu.com/p/335147713
7.批归一化是为了解决什么问题?它的参数有何意义?它在网络中一般放在什么位置?
https://zhuanlan.zhihu.com/p/93643523#:~:text=Batch%20Normalization%E6%98%AF2015%E5%B9%B4%E4%B8%80%E7%AF%87%E8%AE%BA%E6%96%87%E4%B8%AD%E6%8F%90%E5%87%BA%E7%9A%84%E6%95%B0%E6%8D%AE%E5%BD%92%E4%B8%80%E5%8C%96%E6%96%B9%E6%B3%95%EF%BC%8C%E5%BE%80%E5%BE%80%E7%94%A8%E5%9C%A8%E6%B7%B1%E5%BA%A6%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%AD%E6%BF%80%E6%B4%BB%E5%B1%82%E4%B9%8B%E5%89%8D%E3%80%82,%E5%85%B6%E4%BD%9C%E7%94%A8%E5%8F%AF%E4%BB%A5%E5%8A%A0%E5%BF%AB%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83%E6%97%B6%E7%9A%84%E6%94%B6%E6%95%9B%E9%80%9F%E5%BA%A6%EF%BC%8C%E4%BD%BF%E5%BE%97%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83%E8%BF%87%E7%A8%8B%E6%9B%B4%E5%8A%A0%E7%A8%B3%E5%AE%9A%EF%BC%8C%E9%81%BF%E5%85%8D%E6%A2%AF%E5%BA%A6%E7%88%86%E7%82%B8%E6%88%96%E8%80%85%E6%A2%AF%E5%BA%A6%E6%B6%88%E5%A4%B1%E3%80%82%20%E5%B9%B6%E4%B8%94%E8%B5%B7%E5%88%B0%E4%B8%80%E5%AE%9A%E7%9A%84%E6%AD%A3%E5%88%99%E5%8C%96%E4%BD%9C%E7%94%A8%EF%BC%8C%E5%87%A0%E4%B9%8E%E4%BB%A3%E6%9B%BF%E4%BA%86Dropout%E3%80%82
8.用于分类任务的卷积神经网络的最后几层一般是什么层?在最近几年有什么变化?
https://www.zhihu.com/question/365926784
https://blog.csdn.net/u012905422/article/details/52463324
https://zhuanlan.zhihu.com/p/75056190
9.简述卷积神经网络近年来在结构设计上的主要发展和变迁(从AlexNet 到 ResNet 系列)。
https://zhuanlan.zhihu.com/p/378408695
10.卷积神经网络中的瓶颈结构和沙漏结构提出的初衷是什么?可以应用于哪些问题?
https://zhuanlan.zhihu.com/p/378408695
https://www.cnblogs.com/cyssmile/p/13570042.html
第2章 循环神经网络
11.Dropout 为什么可以缓解过拟合问题?
https://blog.csdn.net/qq_19672707/article/details/88740832
https://www.zhihu.com/question/557949986
12.描述循环神经网络的结构及参数更新方式？如何使用卷积神经网络对序列数据建模?
https://zhuanlan.zhihu.com/p/123211148
https://www.jianshu.com/p/39a99c88a565
https://www.jianshu.com/p/247a72812aff
13.循环神经网络为什么容易出现长期依赖问题?
https://zhuanlan.zhihu.com/p/404790442#:~:text=%E9%95%BF%E6%9C%9F%E4%BE%9D%E8%B5%96%E9%97%AE%E9%A2%98%EF%BC%881%E3%80%81%E7%BD%91%E7%BB%9C%E5%B1%82%E6%95%B0%E5%A2%9E%E5%A4%A7%EF%BC%8C%E8%AF%AF%E5%B7%AE%2F%E6%A2%AF%E5%BA%A6%E5%AE%B9%E6%98%93%E6%B6%88%E5%A4%B1%2F%E7%88%86%E7%82%B8%EF%BC%8C%E8%BF%9B%E8%80%8C%E4%BC%98%E5%8C%96%E5%9B%B0%E9%9A%BE%EF%BC%9B2%E3%80%81%E8%BE%93%E5%85%A5%E5%BA%8F%E5%88%97%E8%B6%8A%E9%95%BF%EF%BC%8C%E7%9B%B8%E5%BD%93%E4%BA%8E%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84%E8%B6%8A%E6%B7%B1%EF%BC%8C%E8%B6%8A%E5%AE%B9%E6%98%93%E5%87%BA%E7%8E%B0%E9%95%BF%E6%9C%9F%E4%BE%9D%E8%B5%96%E9%97%AE%E9%A2%98%EF%BC%9B3%E3%80%81%E5%8E%9F%E5%9B%A0%EF%BC%9A%E9%87%8D%E5%A4%8D%E4%BD%BF%E7%94%A8%E7%9B%B8%E5%90%8C%E5%BE%AA%E7%8E%AF%E6%A8%A1%E5%9D%97%EF%BC%8C%E5%AF%BC%E8%87%B4%E4%BF%A1%E6%81%AF%E7%9A%84%E5%89%8D%E5%90%91%E4%BC%A0%E6%92%AD%E5%92%8C%E8%AF%AF%E5%B7%AE%E5%8F%8D%E5%90%91%E4%BC%A0%E6%92%AD%E9%83%BD%E4%BC%9A%E5%87%BA%E7%8E%B0%E5%90%8C%E4%B8%80%E4%B8%AA%E7%9F%A9%E9%98%B5%E7%9A%84%E5%B9%82%EF%BC%8C%E5%AE%B9%E6%98%93%E5%87%BA%E7%8E%B0%E8%AF%AF%E5%B7%AE%2F%E6%A2%AF%E5%BA%A6%E5%AE%B9%E6%98%93%E6%B6%88%E5%A4%B1%2F%E7%88%86%E7%82%B8%EF%BC%9B4%E3%80%81%E8%A7%A3%E5%86%B3%EF%BC%9A%E6%AD%A3%E5%88%99%E5%8C%96%E7%AD%89%EF%BC%9B%E6%97%B6%E9%97%B4%E4%B8%8A%E6%B7%BB%E5%8A%A0%E8%B7%B3%E8%B7%83%E8%BF%9E%E6%8E%A5%EF%BC%9B%E9%95%BF%E7%9F%AD%E6%9C%9F%E8%AE%B0%E5%BF%86%E7%BD%91%E7%BB%9C,%28LSTM%29%E5%92%8C%E9%97%A8%E6%8E%A7%E5%BE%AA%E7%8E%AF%E5%8D%95%E5%85%83%20%28GRU%29%E7%AD%89%E6%96%B0%E5%9E%8B%E7%BD%91%E7%BB%9C%E6%9E

  ........
