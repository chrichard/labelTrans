# 0.Instruction
This Repos contains how to transfer label format between yolo, XML, coco, CityScape,kitti ...

# 1.XML to kitti 
use xml2kitti.py
you should replace 
base_xml_dir="your xml file address"
kitti_saved_dir="your output file address"

# 2.XML to yolo 
use xml2yolo.py
you should replace 
xml_files1="your xml file address"
save_txt_files1="your output label file address"

# 3.VIA to coco  & labelme
本项目有以下几个功能：
1、把VIA标注软件导出的json格式文件转换成COCO标注格式的程序
这个程序修改自这里： https://github.com/codingwolfman/VIA2COCO 
非常感谢，因为原作者的代码里面有点bug，所以直接使用是不行的。不过他并未对其进行维护。
所以我修改了一下，亲测可以把VIA导出的json格式的文件转成COCO格式。
测试平台：mmdetection。我在上面进行训练我的模型。他有标准的接口来读取COCO格式的标注文件。如果格式不正确会报错。亲测可以成功进行训练并得到满意的结果。进行了两个任务：分类和实例分割，bbox和segmentation检测是没有问题的。

2、将VIA标注的文件转为labelme格式

3、检验并显示coco格式的bbox和mask的程序。
转换完成之后，可以用这个程序跑一遍，检测一下是否正确在图片上显示并保存bbox和mask。一般如果能正常跑完。说明你的数据集标注是正确的。如果有错的话，在上一步转换阶段就会报错。

# 6.clear invalid files 
delete the image file which cannot match the label file.
use clearfile.py
you should replace 
input_lbl_dir="your xml file address"
input_img_dir="your output label file address"

# 7.generate val dateset  
split files into val and train dateset .
use generate_val_dataset.py
you should get the param from args.

### Important Note:
   You should modify this address in sourcecode 

