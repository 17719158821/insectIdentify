import os
from shutil import rmtree, copy
import random

# 如果文件存在保证先删除然后再创建
def mk_file(file_path):
    if os.path.exists(file_path):
        rmtree(file_path)
    os.mkdir(file_path)


if __name__ == '__main__':
    # 保证可复现
    random.seed(42)
    # 获取当前地址
    cwd = os.getcwd()
    # 取总数的10%作为验证集
    splitrate = 0.2
    origin_insect_path = os.path.join(cwd, 'dataset\\images')
    insect_class = [insect for insect in os.listdir(origin_insect_path)
                    if os.path.isdir(os.path.join(origin_insect_path, insect))]
    print(len(insect_class))
    # 新建train和val的文件夹
    train_root = os.path.join(cwd, 'dataset\\train')
    val_root = os.path.join(cwd, 'dataset\\val')
    mk_file(val_root)
    mk_file(train_root)
    for insect in insect_class:
        mk_file(os.path.join(train_root, insect))
        mk_file(os.path.join(val_root, insect))

    # 通过random.sample来获取总数0.1的图片名称进行划分
    for insect in insect_class:
        insect_path = os.path.join(origin_insect_path, insect)
        images = os.listdir(insect_path)
        number_images = len(images)
        eval_index = random.sample(images, int(splitrate * number_images))
        for index, image in enumerate(images):
            if image in eval_index:
                # 存放在验证文件夹
                origin_path = os.path.join(insect_path, image)
                new_path = os.path.join(val_root, insect)
                copy(origin_path, new_path)
            else:
                # 将图片存放在训练文件夹
                origin_path = os.path.join(insect_path, image)
                new_path = os.path.join(train_root, insect)
                copy(origin_path, new_path)
                #  显示数据处理的进度
            print("\r[{}] processing [{}/{}]".format(insect, index + 1, number_images), end="")  # processing bar
        print()
    print('Process Done')
