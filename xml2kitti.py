import xml.etree.ElementTree as ET
import os

base_xml_dir="/tao/tlt-experiments/data/training/annotations/xmls/"
kitti_saved_dir="/tao/tlt-experiments/data/training/labels/"

#base_xml_dir = "annotations/xmls/"
xml_list = os.listdir(base_xml_dir)
#kitti_saved_dir = "labels/"


def convert_annotation(file_name):
    in_file = open(base_xml_dir + file_name)
    tree = ET.parse(in_file)
    root = tree.getroot()

    #with open(kitti_saved_dir + file_name[:-4] + '.txt', 'w') as f:
    for obj in root.iter('object'):

        cls = obj.find('name').text
        
        xmlbox = obj.find('bndbox')
        """
            第5～8这4个数：物体的2维边界框
            xmin，ymin，xmax，ymax
        """
        xmin, ymin, xmax, ymax = xmlbox.find('xmin').text, xmlbox.find('ymin').text, \
                                    xmlbox.find('xmax').text, xmlbox.find('ymax').text
        with open(kitti_saved_dir + file_name[:-4] + '.txt', 'a') as f:
            f.write(cls + " " + '0' + " " + '0' + " " + '0' + " " + str(xmin) + '.0' + " "
                    + str(ymin) + '.0' + " " + str(xmax) + '.0' + " " + str(ymax) + '.0' + " " +
                    '0' + " " + '0' + " " + '0' + " " + '0' + " " + '0' + " " + '0' + " " + '0' + '\n')


for i in xml_list:
    convert_annotation(i)

