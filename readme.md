# 引言

相关链接：
1. [LoRA 微软官方代码链接](https://github.com/microsoft/LoRA)
2. [图解大模型微调系列之：大模型低秩适配器 LoRA（原理篇）](https://zhuanlan.zhihu.com/p/646831196)
3. [图解大模型微调系列之：大模型低秩适配器 LoRA（源码解读与实操篇）](https://zhuanlan.zhihu.com/p/654897296)
4. [本仓库对应的 CSDN 博客](https://blog.csdn.net/BIT_Legend/article/details/139121023?csdn_share_tail=%7B%22type%22%3A%22blog%22%2C%22rType%22%3A%22article%22%2C%22rId%22%3A%22139121023%22%2C%22source%22%3A%22BIT_Legend%22%7D)
   
关于 LoRA 的原理和源码解读，以上博客 2 和 3 写的非常好，强烈建议仔细阅读，博客 4 是对 LoRA 微软官方源码的进一步解读。包含对源码框架的解读、对源码一句一句的汉语注释，以及对源码中训练过程和推理过程的分析。

本仓库是重新整理删除没有使用的函数和类后带注释的 LoRA 源码，为了便于理解，这里打乱了原代码目录结构，所以这里的代码无法运行测试，关于运行测试请使用上面的博客 1 和 3 的方法。