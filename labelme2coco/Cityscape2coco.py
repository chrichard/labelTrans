import os
import argparse
import json

from labelme import utils
import numpy as np
import glob
import PIL.Image

ratio=1 #图像缩小0.5
labelme_images = '/AI/DLDataSet/cityscape/cityscape_labelme/val2017'

def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()

class labelme2coco(object):
    def __init__(self, labelme_json=[], save_json_path="./coco.json"):
        """
        :param labelme_json: the list of all labelme json file paths
        :param save_json_path: the path to save new json
        """
        self.labelme_json = labelme_json
        self.save_json_path = save_json_path
        self.images = []
        self.categories = []
        self.annotations = []
        
        self.label = [["bicycle"],[ "bicyclegroup"],[ "bridge"],["building"],["bus"],["busgroup"],[ "car"],[ "caravan"],[ "cargroup"],["ego vehicle"],
                                ["fence"],["ground"],[ "guard rail"],[ "motorcycle"],["motorcyclegroup"],[ "out of roi"],[ "parking"],[ "person"],[ "persongroup"],
                                ["pole"],["rail track"],[ "rectification border"],[ "rider"],[ "ridergroup"],[ "road"],[ "sidewalk"],[ "sky"],[ "terrain"],["traffic light"],
                                ["traffic sign"],["trailer"],[ "train"],["traingroup"],["truck"],["truckgroup"],[ "tunnel"],[ "vegetation"],["wall"],
                                ["dynamic"],["license plate"],[ "polegroup"],["static"]]#1+38+4
        
        #self.label=[]
        self.annID = 1
        self.height = 0
        self.width = 0

        self.save_json()

    def data_transfer(self):
        for num, json_file in enumerate(self.labelme_json):
            with open(json_file, "r") as fp:
                print(num," of  ",len(labelme_json))
                data = json.load(fp)

                ##判断点数是否过多。如果过多不保存
                toltal_point = 0
                shapeidx = 0
                for shapes in data["objects"]:
                    shapeidx+=1
                    points = shapes["polygon"]
                    toltal_point += len(points)

                filename =  json_file.split("/")[-1]
                file_pre = filename[:3]

                if  ((toltal_point>2500) and (file_pre=="ham") ):
                    print(num," :  ",json_file,"  -- too many toltal_point : ",toltal_point)
                    continue

                if  ((toltal_point>3000) and (file_pre=="han") ):
                    print(num," :  ",json_file,"  -- too many toltal_point : ",toltal_point)
                    continue

                if  ((toltal_point>3000) and (file_pre=="stu") ):
                    print(num," :  ",json_file,"  -- too many toltal_point : ",toltal_point)
                    continue

                if  ((toltal_point>3100) and (file_pre=="dar") ):
                    print(num," :  ",json_file,"  -- too many toltal_point : ",toltal_point)
                    continue
                ############################
                self.images.append(self.image(data, num,json_file))

                for shapes in data["objects"]:

                    label = shapes["label"].split("_")
                    if label not in self.label:
                        self.label.append(label)
                    points = shapes["polygon"]
                    toltal_point += len(points)
                    #if (len(points)>1000):#ceshi
                        #print(num," :  ",json_file,"  -- too many points : ",len(points))
                        #continue

                    for point in points:
                        point[0]=(int)(point[0]*ratio)
                        point[1]=(int)(point[1]*ratio)

                    self.annotations.append(self.annotation(points, label, num))
                    self.annID += 1
                 
                #if  (shapeidx>100):
                #    print(num," :  ",json_file,"  -- too many shapes : ",shapeidx)

        # Sort all text labels so they are in the same order across data splits.
        # self.label.sort()
        for label in self.label:
            self.categories.append(self.category(label))
        for annotation in self.annotations:
            annotation["category_id"] = self.getcatid(annotation["category_id"])

    def image(self, data, num,json_file):
        image = {}
        self.height= (int)(data["imgHeight"]*ratio)
        self.width = (int)(data["imgWidth"]*ratio)
        image["height"] =  self.height
        image["width"] =  self.width
        image["id"] = num

        imgfile =  json_file.split("/")[-1]
        imgfile = imgfile.split("_gtFine_polygons.json")[0]+"_leftImg8bit.png"
        #读取图片，然后缩放
        img_org= PIL.Image.open(labelme_images+"/"+imgfile)
        img_resize=img_org.resize(( self.width, self.height,))

        imgfile_resize = imgfile.split("_gtFine_polygons.json")[0]+".jpg"
        img_resize.save(labelme_images+"/"+imgfile_resize)
        image["file_name"] =imgfile_resize# json_file.split("/")[-1]

        return image

    def category(self, label):
        category = {}
        category["supercategory"] = label[0]
        category["id"] = len(self.categories)+1
        category["name"] = label[0]
        return category

    def annotation(self, points, label, num):
        annotation = {}
        contour = np.array(points)
        x = contour[:, 0]
        y = contour[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        annotation["segmentation"] = [list(np.asarray(points).flatten())]
        annotation["iscrowd"] = 0
        annotation["area"] = area
        annotation["image_id"] = num

        annotation["bbox"] = list(map(float, self.getbbox(points)))

        annotation["category_id"] = label[0]  # self.getcatid(label)
        annotation["id"] = self.annID
        return annotation

    def getcatid(self, label):
        for category in self.categories:
            if label == category["name"]:
                return category["id"]
        print("label: {} not in categories: {}.".format(label, self.categories))
        exit()
        return -1

    def getbbox(self, points):
        polygons = points
        mask = self.polygons_to_mask([self.height, self.width], polygons)
        return self.mask2box(mask)

    def mask2box(self, mask):

        index = np.argwhere(mask == 1)
        rows = index[:, 0]
        clos = index[:, 1]

        left_top_r = np.min(rows)  # y
        left_top_c = np.min(clos)  # x

        right_bottom_r = np.max(rows)
        right_bottom_c = np.max(clos)

        return [
            left_top_c,
            left_top_r,
            right_bottom_c - left_top_c,
            right_bottom_r - left_top_r,
        ]

    def polygons_to_mask(self, img_shape, polygons):
        mask = np.zeros(img_shape, dtype=np.uint8)
        mask = PIL.Image.fromarray(mask)
        xy = list(map(tuple, polygons))
        PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
        mask = np.array(mask, dtype=bool)
        return mask

    def data2coco(self):
        data_coco = {}
        data_coco["images"] = self.images
        data_coco["categories"] = self.categories
        data_coco["annotations"] = self.annotations
        return data_coco



    def save_json(self):
        print("save coco json")
        self.data_transfer()
        self.data_coco = self.data2coco()

        print(self.save_json_path)
        os.makedirs(
            os.path.dirname(os.path.abspath(self.save_json_path)), exist_ok=True
        )
        json.dump(self.data_coco, open(self.save_json_path, "w"), indent=4,default=np_encoder)


if __name__ == "__main__":
    import argparse
    '''
    parser = argparse.ArgumentParser(
        description="labelme annotation to coco data json file."
    )
    parser.add_argument(
        "labelme_images",
        help="Directory to labelme images and annotation json files.",
        type=str,
    )
    parser.add_argument(
        "--output", help="Output json file path.", default="trainval.json"
    )
    args = parser.parse_args()
    labelme_json = glob.glob(os.path.join(args.labelme_images, "*.json"))
    '''
    #labelme_images = 'images'
    labelme_json =glob.glob(os.path.join(labelme_images, "*.json"))
    

    '''
    labelme_json = glob.glob(os.path.join(labelme_images, "a*.json"))
    labelme_json +=glob.glob(os.path.join(labelme_images, "b*.json"))
    labelme_json +=glob.glob(os.path.join(labelme_images, "c*.json"))
    #labelme_json +=glob.glob(os.path.join(labelme_images, "d*.json"))
    labelme_json +=glob.glob(os.path.join(labelme_images, "e*.json"))
    #labelme_json +=glob.glob(os.path.join(labelme_images, "h*.json"))
    labelme_json +=glob.glob(os.path.join(labelme_images, "j*.json"))
    labelme_json +=glob.glob(os.path.join(labelme_images, "k*.json"))
    labelme_json +=glob.glob(os.path.join(labelme_images, "m*.json"))
    #labelme_json +=glob.glob(os.path.join(labelme_images, "s*.json"))

    labelme_json +=glob.glob(os.path.join(labelme_images, "t*.json"))
    
    #labelme_json +=glob.glob(os.path.join(labelme_images, "u*.json"))
    #labelme_json +=glob.glob(os.path.join(labelme_images, "w*.json"))
    #labelme_json +=glob.glob(os.path.join(labelme_images, "z*.json"))
    '''
    output='/AI/DLDataSet/cityscape/cityscape_labelme/annotations/instances_val2017.json'
    labelme2coco(labelme_json, output)
