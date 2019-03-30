# camera_face_check
基于opencv，tensorflow，调用摄像头的人脸检测
csdn博客链接：https://blog.csdn.net/weixin_43582101/article/details/88913164



![在这里插入图片描述](https://img-blog.csdnimg.cn/20190330145449955.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzU4MjEwMQ==,size_16,color_FFFFFF,t_70)
直接贴代码了，基本上都有注释。就不多说了。

导入要使用的模块 cv2，tensorflow
    
    import tensorflow as tf
    from face_check import detect_face
    import cv2
    import numpy as np

tf.Graph() 表示实例化了一个用于 tensorflow 计算和表示用的数据流图
   
先用两个人物测试。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190330150619887.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzU4MjEwMQ==,size_16,color_FFFFFF,t_70)
真人测试。拒绝露脸![在这里插入图片描述](https://img-blog.csdnimg.cn/20190330150708123.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzU4MjEwMQ==,size_16,color_FFFFFF,t_70)

