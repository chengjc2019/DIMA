# -*- coding: utf-8 -*-
import argparse
import mmcv
# from mmdet.datasets import build_dataset
from mmrotate.datasets import build_dataset
from mmcv import Config
import os
import DOTA_devkit.dota_utils as util
import tqdm
from xml.dom.minidom import Document, parse


# import locale
# locale.setlocale(locale.IC_ALL, 'en')

def parse_args():
    """Parse parameters."""
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    # parser.add_argument('--config', default='/home/lyj/OBBDetection-master/configs/'
    #                                         'obb/roi_transformer/faster_rcnn_roitrans_r50_fpn_1x_dota10.py')
    parser.add_argument('--config',
                        default='/disk1/cjc/codes/mmrotate/configs/oriented_rcnn/oriented_rcnn_r50_fpn_1x_dota_le90.py')
    parser.add_argument('--pkl', default=' ')
    parser.add_argument('--output', default=' ')
    parser.add_argument('--sub', default=' ')

    args = parser.parse_args()

    return args


def Task1_results2txt(srcpath, dstpath):
    dstname = os.path.join(dstpath, 'result.txt')

    filelist = util.GetFileFromThisRootDir(srcpath)
    count_class = {}
    with open(dstname, 'w') as f_out:
        for fullname in filelist:
            name = util.custombasename(fullname)
            if name not in count_class:
                count_class[name] = 0
            with open(fullname, 'r', encoding='utf-8', errors='ignore') as f_in:
                lines = f_in.readlines()
                for x in lines:
                    count_class[name] += 1
                    x = x.strip('\n')
                    parts = [x, name]
                    y = ' '.join(parts)
                    f_out.write(y + '\n')
    print(count_class)


def txt2singletxt(filename, dstname):
    if os.path.exists(dstname) == False:
        os.mkdir(dstname)

    with open(filename, 'r') as f_in:
        lines = f_in.readlines()
        nameboxdict = {}
        splitlines = [x.strip().split(' ') for x in lines]
        for splitline in splitlines:
            image_name = splitline[0]
            det = splitline[1:]
            if (image_name not in nameboxdict):
                nameboxdict[image_name] = []
            det = list(det)
            # print("det:", det)
            nameboxdict[image_name].append(det)
            # print(image_name)

    result_name = list(nameboxdict.keys())
    result_context = list(nameboxdict.values())
    # print("result_name:",  result_name)
    result_list = []
    for index in range(len(nameboxdict)):
        txt_name = os.path.join(dstname, result_name[index], ) + '.txt'
        # print("result_name[index]:", type(txt_name))
        # print(index)
        with open(txt_name, 'w', encoding="utf-8") as f_out:
            for i in range(len(result_context[index])):
                a = result_context[index]
                # b =a[i][1:]#no score
                b = a[i]
                outline = ' '.join(b)
                f_out.write(outline + '\n')


def add_none_txt(path):
    files = os.listdir(path)
    i = 0
    non = []
    for i in range(8137):
        st = str(i) + '.txt'
        if st not in files:
            print(st)
            non.append(st)
            open(os.path.join(path, st), 'w')


def add_obj2objs(doc, objects, class_name, bbox, score):
    obj = doc.createElement('object')
    coord = doc.createElement('coordinate')
    coord_txt = doc.createTextNode('pixel')
    coord.appendChild(coord_txt)
    obj.appendChild(coord)

    type_obj = doc.createElement('type')
    type_obj_txt = doc.createTextNode('rectangle')
    type_obj.appendChild(type_obj_txt)
    obj.appendChild(type_obj)

    description = doc.createElement('description')
    description_txt = doc.createTextNode('None')
    description.appendChild(description_txt)
    obj.appendChild(description)

    possibleresult = doc.createElement('possibleresult')
    name = doc.createElement('name')
    name_txt = doc.createTextNode(class_name)
    name.appendChild(name_txt)
    possibleresult.appendChild(name)

    probability = doc.createElement('probability')
    probability_txt = doc.createTextNode(str(score))
    probability.appendChild(probability_txt)
    possibleresult.appendChild(probability)
    obj.appendChild(possibleresult)

    points = doc.createElement('points')
    for i in range(4):
        point = doc.createElement('point')
        point_txt = doc.createTextNode(str(bbox[2 * i]) + ',' + str(bbox[2 * i + 1]))
        point.appendChild(point_txt)
        points.appendChild(point)
    point = doc.createElement('point')
    point_txt = doc.createTextNode(str(bbox[0]) + ',' + str(bbox[1]))
    point.appendChild(point_txt)
    points.appendChild(point)
    obj.appendChild(points)
    objects.appendChild(obj)
    return objects


def txt2xml(txt_path, dst_path):
    txt_list = os.listdir(txt_path)
    txt_list = tqdm.tqdm(txt_list)
    for txt_idx in txt_list:
        txt_file = open(os.path.join(txt_path, txt_idx))
        txt_lines = txt_file.readlines()
        if txt_lines:
            init_tree = parse("/home/chip/mmrotate/submit_program/GF_benchmark_pre/gf_model.xml")
            doc = Document()
            root = init_tree.documentElement
            # objects
            root_objs = root.getElementsByTagName('objects')[0]
            # img_name
            name = txt_idx.replace('.txt', '.tif')
            filename = root.getElementsByTagName('filename')[0]
            filename.firstChild.data = name
            # xml_name
            dst_file_path = os.path.join(dst_path, txt_idx.replace('.txt', '.xml'))
            #
            for line in txt_lines:
                line_split = line.strip('\n').split(' ')
                if len(line_split) == 0:
                    continue
                else:
                    score = line_split[0][:6]
                    coords = line_split[1:-1]
                    coords = [int(float(coor_i)) for coor_i in coords]
                    if line_split[-1] == 'Passenger-Ship':
                        obj_name = 'Passenger Ship'
                    elif line_split[-1] == 'Fishing-Boat':
                        obj_name = 'Fishing Boat'
                    elif line_split[-1] == 'Engineering-Ship':
                        obj_name = 'Engineering Ship'
                    elif line_split[-1] == 'Liquid-Cargo-Ship':
                        obj_name = 'Liquid Cargo Ship'
                    elif line_split[-1] == 'Dry-Cargo-Ship':
                        obj_name = 'Dry Cargo Ship'
                    elif line_split[-1] == 'Small-Car':
                        obj_name = 'Small Car'
                    elif line_split[-1] == 'Cargo-Truck':
                        obj_name = 'Cargo Truck'
                    elif line_split[-1] == 'Dump-Truck':
                        obj_name = 'Dump Truck'
                    elif line_split[-1] == 'Truck-Tractor':
                        obj_name = 'Truck Tractor'
                    elif line_split[-1] == 'Basketball-Court':
                        obj_name = 'Basketball Court'
                    elif line_split[-1] == 'Tennis-Court':
                        obj_name = 'Tennis Court'
                    elif line_split[-1] == 'Football-Field':
                        obj_name = 'Football Field'
                    elif line_split[-1] == 'Baseball-Field':
                        obj_name = 'Baseball Field'
                    else:
                        obj_name = line_split[-1]
                    # obj_name = line_split[-1].replace('-', ' ')
                    root_objs = add_obj2objs(doc, root_objs, obj_name, coords, score)
            with open(dst_file_path, 'wb') as f:
                f.write(init_tree.toprettyxml(indent='\t', newl="\n", encoding='utf-8'))
                f.close()
        else:
            dst_file_path = os.path.join(dst_path, txt_idx.replace('.txt', '.xml'))
            init_tree = parse('/home/chip/mmrotate/submit_program/GF_benchmark_pre/gf_model_1.xml')
            doc = Document()
            root = init_tree.documentElement
            # img_name
            name = txt_idx.replace('.txt', '.tif')
            filename = root.getElementsByTagName('filename')[0]
            filename.firstChild.data = name
            #
            with open(dst_file_path, 'wb') as f:
                f.write(init_tree.toprettyxml(indent='\t', newl="\n", encoding='utf-8'))
                f.close()


if __name__ == '__main__':
    # error_pkl ='/data2/lyj/SH/ReDet/work_dirs/mmrotate/error_04_07_r50_module1_1x_gaofen_ms_2GPU/debug/12.pkl'
    # error_pkl ='/data1/lyj/checkpoint/work_dirs/obb/05_24_33conv_128_module2_ms_weight_05_4/benchmark_ss/12.pkl'

    # for obb
    # pkl_file = '/data1/cjc/OBBDetection/work_dirs/debug/0526.pkl'
    # output_path = '/data1/cjc/OBBDetection/work_dirs/debug/benchmark_ss/'
    # sub_path = '/data1/cjc/OBBDetection/work_dirs/debug/benchmark_ss/sub/'

    args = parse_args()
    pkl_file = args.pkl
    output_path = args.output
    sub_path = args.sub
    #
    type = 'OBB'
    # parse_results(config_file, pkl_file, output_path, type)

    outputs = mmcv.load(pkl_file)
    # baseline_outpust =mmcv.load(error_pkl)
    cfg = Config.fromfile(args.config)
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    with_merge = True
    ign_scale_ranges = None
    iou_thr = 0.5
    nproc = 4
    save_dir = sub_path
    result_files, tmp_dir = dataset.format_results(outputs, save_dir)
    Task1_results2txt(sub_path, output_path)

    filename = os.path.join(output_path, 'result.txt')
    dstname = os.path.join(output_path, 'txt/')
    txt2singletxt(filename, dstname)
    add_none_txt(dstname)
    final_test_path = os.path.join(output_path, 'test/')
    if not os.path.exists(final_test_path):
        os.mkdir(final_test_path)
    txt2xml(dstname, final_test_path)

    # cd = 'cd /data1/cjc/OBBDetection/work_dirs/debug/benchmark_ss/ '
    # print('\n' + cd)
    # # os.system(cd)
    #
    # cmd = 'zip -rq test.zip test/'
    # print('\n' + cmd)
    # print('end!')
