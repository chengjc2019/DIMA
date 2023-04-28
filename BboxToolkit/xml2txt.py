from xml.dom.minidom import parse
import os


# 将xml格式的点标注文件转换为txt格式
# (可以一个xml文件里面包含所有的图片标注信息，也可以一个xml对应1张图片的标签信息)

# 将每一个xml文件转换为txt格式:(x,y)
def convert_annotation(xmlfilepath, txtpath):
    in_file = open(xmlfilepath, encoding='utf-8')

    tree = parse(in_file)
    root = tree.documentElement

    # 获取所有images
    images = root.getElementsByTagName("image")
    for image in images:
        imgname = image.getAttribute("name").split(".")[0]
        out_file = open(txtpath + "/" + '%s.txt' % (imgname), 'a', encoding='utf-8')  # 生成txt格式文件
        points = image.getElementsByTagName("points")
        for point in points:
            if point.hasAttribute("points"):
                point = str(point.getAttribute("points"))
                x = point.split(",")[0]
                y = point.split(",")[1]
                out_file.write(x + " " + y + '\n')
        out_file.close()


if __name__ == '__main__':
    # xml文件夹
    xmlpath = r'/home/chip/obbstacking/work_dirs/orcnn_swin_trainval+redet_trainval/result_root_dir/ensemble_result/test'
    # txt文件夹
    txtpath = r'/home/chip/obbstacking/work_dirs/orcnn_swin_trainval+redet_trainval/result_root_dir/ensemble_result/labelTxt'
    # 读取xml文件夹下的所有xml文件
    filelist = os.listdir(xmlpath)
    for files in filelist:
        # 获取每一个xml文件名
        xmlfilepath = xmlpath + "/" + files
        # 将每一个xml文件转换为txt
        convert_annotation(xmlfilepath, txtpath)
