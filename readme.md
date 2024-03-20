1.克隆仓库 (Clone the Repository)
用户需要从GitHub或其他代码托管平台克隆项目到本地。
2.安装
本机使用的python版本为3.10
运行如下命令安装依赖包
pip install -r requirements.txt
3.如何使用
如要体验训练到测试的全过程，安装完依赖包后打开main.py文件，运行即可输出模型的准确率等各项指标，也可在代码末尾add_argument里面修改default的值来改变训练集和测试集路径，包括模型学习率等各种模型的参数。
修改参数的代码示例如
parser.add_argument('--batch_size', type=int, help='批量Batch size for training'，default=8000 )
只需修改default中的8000即可。
4.直接测试与预测
为了方便快速得到结果，我们提供了直接使用已有的模型进行测试和预测的代码，可以打开test_predict文件夹，安装好环境后后运行test.py即可输出准确率，召回率等各项指标。或者选择predict.py文件，运行后会在该文件夹内输出预测结果的csv文件。
5.运行环境
本机的训练是在3090的GPU上运行。