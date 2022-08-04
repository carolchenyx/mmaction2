import os

path = '/media/hkuit155/carol/Dataset/ucf-crime/Anomaly-Videos/Normal/'

folderList=os.listdir(path)

n = 0
for i in folderList:
    # 设置旧文件名（就是路径+文件名）
    path_filelist = path + folderList[n] + os.sep  # os.sep添加系统分隔符
    filelist = os.listdir(path_filelist)
    m = 0
    for j in filelist:
        oldname = path_filelist + filelist[m]
        # 设置新文件名
        newname = path_filelist + 'img_' + filelist[m]
        m +=1
        os.rename(oldname, newname)  # 用os模块中的rename方法对文件改名
        print(oldname, '======>', newname)

    n += 1