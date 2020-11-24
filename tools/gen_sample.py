import os
import random
import cv2

video_path = r'D:\fight-tsn-tsm\tools\video'
frames_path = r'D:\fight-tsn-tsm\tools\frames'
splits_path = r'D:\fight-tsn-tsm\tools\splits'

train_data = []
val_data = []
test_data = []

for sub_dir in os.listdir(video_path):
    frames_sub_dir = os.path.join(frames_path, sub_dir)
    print(frames_sub_dir)
    if not os.path.exists(frames_sub_dir):
        os.mkdir(frames_sub_dir)

    # 制作标签
    tag_list = []
    video_sub_dir = os.path.join(video_path, sub_dir)
    for filename in os.listdir(video_sub_dir):
        name = filename.split(".")[0]
        tag = sub_dir + "," + name
        tag_list.append(tag)

        video_last_path = os.path.join(video_sub_dir, filename)
        frames_last_path = os.path.join(frames_sub_dir, name)
        if not os.path.exists(frames_last_path):
            os.mkdir(frames_last_path)
        print(video_last_path)
        print(frames_last_path)
        cap = cv2.VideoCapture(video_last_path)
        i_frame = 0
        while True:
            i_frame += 1
            flag, img = cap.read()
            if not flag:
                break
            # cv2.imshow('frame', img)
            print(os.path.join(frames_last_path, str(i_frame).zfill(6) + ".jpg"))
            cv2.imwrite(os.path.join(frames_last_path, str(i_frame).zfill(6) + ".jpg"), img)
            # cv2.waitKey(500)
        cap.release()
        cv2.destroyAllWindows()

    # 打乱顺序
    random.shuffle(tag_list)
    random.shuffle(tag_list)
    random.shuffle(tag_list)

    # 划分训练集:验证集:测试集=7:1.5:1.5
    leng = len(tag_list)
    train_data.extend(tag_list[:int(leng * 0.7)])
    val_data.extend(tag_list[int(leng * 0.7):int(leng * 0.85)])
    test_data.extend(tag_list[int(leng * 0.85):])

print(len(train_data), len(val_data), len(test_data))

with open(os.path.join(splits_path, 'train.csv'), mode='w') as train_file:
    train_file.write('label,name\n')
    for e in train_data:
        train_file.write(e)
        train_file.write('\n')
    train_file.close()
with open(os.path.join(splits_path, 'val.csv'), mode='w') as val_file:
    val_file.write('label,name\n')
    for e in val_data:
        val_file.write(e)
        val_file.write('\n')
    val_file.close()
with open(os.path.join(splits_path, 'test.csv'), mode='w') as test_file:
    test_file.write('label,name\n')
    for e in test_data:
        test_file.write(e)
        test_file.write('\n')
    test_file.close()
