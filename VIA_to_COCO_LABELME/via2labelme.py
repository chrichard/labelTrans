import json
import numpy as np
from PIL import Image
import base64

path='/tao/maskrcnn/data/raw-data/validating_data/via_region_data_rider.json'  #via生成的json
imgpath='/tao/maskrcnn/data/raw-data/validating_data/' #图片目录

#path='/tao/maskrcnn/data/raw-data/training_data/via_region_data_rider.json'  #via生成的json
#imgpath='/tao/maskrcnn/data/raw-data/training_data/' #图片目录
with open(path,'r',encoding='utf8')as fp:    #打开json文件
    json_data = json.load(fp)
    for count,L in enumerate(json_data):
        full_filename = json_data[L]['filename']
        print(full_filename)
        filename=full_filename[0:-4]  
                 #由于via标注的时候是所有图片的标签集中在一个json文件，我的目标是一个图片生成一个
                 #所以要把每个图片的名字保存下来
        region=json_data[L]['regions']
        shapes=[]
        group_id = 0
        for K in region:
            reg=K['shape_attributes']       
            point_X=reg['all_points_x']
            point_Y=reg['all_points_y']
            resultpos=list(zip(point_X,point_Y))
            resultpos=np.array(resultpos)
            resultpos=resultpos.tolist()  
            #为了跟labelme中的坐标相同， 对xy坐标先合成，转为numpy再转会List
            group_id+=1
            dict1={"label":"lane","points":resultpos,"group_id":group_id}
            shapes.append(dict1)
        print('test=',shapes)
        #获取图片的信息

        img = Image.open(imgpath+full_filename)

        width = img.size[0]  # 宽
        height = img.size[1] # 高
        with open(imgpath+full_filename,'rb') as f:
                img_data = f.read()
                base64_data = base64.b64encode(img_data)
                base64_str = str(base64_data, 'utf-8') 

        one = {"imagePath": full_filename,
                    "version": "5.0.1",
                    "imageData": base64_str,
                     "imgWidth": width,
                    "imgHeight": height,
                     "shapes":shapes}
        with open('res/'+filename+'.json','w',encoding='utf-8') as f:
            f.write(json.dumps(one,ensure_ascii=False,indent=1)) #最后保持

