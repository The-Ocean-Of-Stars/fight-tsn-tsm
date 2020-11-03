# with open("train.csv", mode="w") as new_f:
#     with open("train_old.csv", mode="r") as old_f:
#         lines = old_f.readlines()
#         # print(len(lines))
#         for i, line in enumerate(lines):
#             if i > 0:
#                 line = line.replace("d.avi,train", "")
#             # print(line)
#             new_f.write(line)
#         old_f.close()
#     new_f.close()
#
# with open("val.csv", mode="w") as new_f:
#     with open("val_old.csv", mode="r") as old_f:
#         lines = old_f.readlines()
#         # print(len(lines))
#         for i, line in enumerate(lines):
#             if i > 0:
#                 line = line.replace("d.avi,val", "")
#             # print(line)
#             new_f.write(line)
#         old_f.close()
#     new_f.close()

# import os
# cmd = 'ffmpeg -i \"{}/{}\" -threads 1 -vf scale=-1:256 -q:v 0 \"{}/{}/%06d.jpg\"'.format(VIDEO_ROOT, video,
#                                                                                              FRAME_ROOT, video[:-5])
# os.system(cmd)
# with open("train_old.csv", "r") as f:
#     lines = f.readlines()[1:]
#     f.close()
