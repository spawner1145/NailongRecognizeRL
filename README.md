# NailongRecognizeRL

我之前的[奶龙检测](https://github.com/spawner1145/NailongRecognize.git)的改进版本

使用了强化学习的思想，并优化了一些预处理操作

这个版本不需要准备test集了，直接放进train相关文件夹就行

run.py和run_rl.py(一个类似强化学习机制的推理脚本)支持外部base64调用，示例在底部

如果单独运行run.py，确保input文件夹中有文件；如果单独运行run_rl.py，确保input_true和input_false文件夹有文件

为了防止占用太多我没放模型，要自己运行一遍train.py以后才可以使用run.py或run_rl.py进行推理

torch和torchvision自己装，没写在requirements.txt里
