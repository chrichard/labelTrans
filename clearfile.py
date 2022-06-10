import os

input_lbl_dir="/AI/DLDataSet/RoadDamageDataset/RoadDamageDataset_all/labels/"
input_img_dir="/AI/DLDataSet/RoadDamageDataset/RoadDamageDataset_all/images/"

def clearimg():
    """Main function for data preparation."""


    img_files = []
    idx = 0
    for file_name in os.listdir(input_img_dir):
        if (file_name.split(".")[-1] == "png") or (file_name.split(".")[-1] == "jpg"):
            lbl_file = file_name.split(".")[0]
            full_lbl_file =input_lbl_dir+lbl_file+".txt"
            if not os.path.exists(full_lbl_file):
                full_img_file =input_img_dir+lbl_file+".jpg"
                print(idx,"   ",full_img_file)
                idx+=1
                os.remove(full_img_file)


def clearlbl():
    """Main function for data preparation."""


    img_files = []
    idx = 0
    for file_name in os.listdir(input_lbl_dir):
        if (file_name.split(".")[-1] == "txt"):
            img_file = file_name.split(".")[0]
            full_img_file =input_img_dir+img_file+".jpg"
            if not os.path.exists(full_img_file):
                full_lbl_file =input_lbl_dir+img_file+".txt"
                print(idx,"   ",full_lbl_file)
                idx+=1
                os.remove(full_lbl_file)

if __name__ == "__main__":
    clearimg()
    clearlbl()