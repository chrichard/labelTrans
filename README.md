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

# 3.clear invalid files 
delete the image file which cannot match the label file.
use clearfile.py
you should replace 
input_lbl_dir="your xml file address"
input_img_dir="your output label file address"

# 4.generate val dateset  
split files into val and train dateset .
use generate_val_dataset.py
you should get the param from args.

### Important Note:
   You should modify this address in sourcecode 

