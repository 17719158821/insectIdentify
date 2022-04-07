import os

# 数据集文件夹重命名
if __name__ == '__main__':
    base_dir = "C:\c_disk\WorkSpace\PyCharm\insectIdentify\dataset\images"
    # print(base_dir)
    dirs = os.listdir(base_dir)
    for (index, data_dir) in enumerate(dirs):
        data_full_dir = os.path.join(base_dir, data_dir)
        print(data_full_dir)
        image_file_list = os.listdir(data_full_dir)
        i = 1
        for ifile in image_file_list:
            old_file_name = os.path.join(data_full_dir, ifile)
            new_file_name = os.path.join(data_full_dir, "{}.jpg".format(i))
            # os.rename(old_file_name, new_file_name)
            i = i + 1

            print(ifile)
