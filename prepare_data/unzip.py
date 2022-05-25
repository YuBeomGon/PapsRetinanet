import os, zipfile

cur_dir = '/'.join(os.path.abspath(__file__).split('/')[:-1])
anno_dir = 'annotations/'

extension = ".zip"
cnt1 = 0
cnt2 = 0
for file in os.listdir(cur_dir+ '/' + anno_dir) :
    print(file)
    if file.endswith('.zip') :
        if not file.startswith('drive-') :
            cnt1 += 1
            d = file.split('-c')[0]
#             print(d)
            zip = zipfile.ZipFile(anno_dir + file)
            if not os.path.isdir(anno_dir + d) :
                cnt2 += 1
                print(d)
                os.mkdir(anno_dir + d)        
                zip.extractall(path=anno_dir + d)
        #         zipfile.ZipFile.extract(item)
            #     if item.endswith(extension):
            #         zipfile.ZipFile.extract(item)
print('total folder {} newly created folder {}'.format(cnt1, cnt2))            