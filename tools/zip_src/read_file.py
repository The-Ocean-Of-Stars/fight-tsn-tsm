import os

# path = './yes'
# yes_delete = []
# for filename in os.listdir(path):
#     yes_delete.append(filename.replace(".mp4", ''))
# print(len(yes_delete), yes_delete)
# print("-------------")
# path = './no'
# no_delete = []
# for filename in os.listdir(path):
#     no_delete.append(filename.replace(".mp4", ''))
# print(len(no_delete), no_delete)


# yes_delete = ['V_1', 'V_17', 'V_2', 'V_222', 'V_236', 'V_237', 'V_272', 'V_276', 'V_277', 'V_278', 'V_283', 'V_286',
#               'V_291', 'V_294', 'V_298', 'V_299', 'V_301', 'V_303', 'V_307', 'V_311', 'V_313', 'V_315', 'V_320',
#               'V_332', 'V_333', 'V_359', 'V_374', 'V_377', 'V_387', 'V_398', 'V_407', 'V_411', 'V_421', 'V_448',
#               'V_641', 'V_642', 'V_643', 'V_644', 'V_645', 'V_646', 'V_647', 'V_7', 'V_750', 'V_751', 'V_759', 'V_760',
#               'V_761', 'V_789', 'V_8', 'V_9']
# yes_tag = 'yes_violence,'
# yes_delete_new = []
# for e in yes_delete:
#     yes_delete_new.append(yes_tag + e)
# print(yes_delete_new)
#
# no_delete = ['NV_1', 'NV_11', 'NV_14', 'NV_17', 'NV_194', 'NV_20', 'NV_21', 'NV_23', 'NV_25', 'NV_26', 'NV_269',
#              'NV_27', 'NV_273', 'NV_274', 'NV_286', 'NV_293', 'NV_295', 'NV_296', 'NV_297', 'NV_298', 'NV_299',
#              'NV_316', 'NV_319', 'NV_32', 'NV_320', 'NV_322', 'NV_353', 'NV_354', 'NV_355', 'NV_356', 'NV_357',
#              'NV_358', 'NV_359', 'NV_360', 'NV_361', 'NV_362', 'NV_363', 'NV_7', 'NV_715', 'NV_716', 'NV_721', 'NV_761',
#              'NV_823', 'NV_824', 'NV_843', 'NV_844', 'NV_846', 'NV_854', 'NV_974', 'NV_992']
# no_tag = 'no_violence,'
# no_delete_new = []
# for e in no_delete:
#     no_delete_new.append(no_tag + e)
# print(no_delete_new)

# # path = './no251_400'
# path = './nv251_400'
# no_delete_more = []
# for filename in os.listdir(path):
#     # no_delete_more.append(filename.replace(".avi", ''))
#     no_delete_more.append(filename.replace(".mp4", ''))
# print(len(no_delete_more), no_delete_more)

# no_delete_more = ['NV_251', 'NV_252', 'NV_253', 'NV_254', 'NV_255', 'NV_256', 'NV_257', 'NV_258', 'NV_259', 'NV_260',
#                   'NV_261', 'NV_262', 'NV_263', 'NV_264', 'NV_265', 'NV_266', 'NV_267', 'NV_268', 'NV_269', 'NV_270',
#                   'NV_271', 'NV_272', 'NV_273', 'NV_274', 'NV_275', 'NV_276', 'NV_277', 'NV_278', 'NV_279', 'NV_280',
#                   'NV_281', 'NV_282', 'NV_283', 'NV_284', 'NV_285', 'NV_286', 'NV_287', 'NV_288', 'NV_289', 'NV_290',
#                   'NV_291', 'NV_292', 'NV_293', 'NV_294', 'NV_295', 'NV_296', 'NV_297', 'NV_298', 'NV_299', 'NV_300',
#                   'NV_301', 'NV_302', 'NV_303', 'NV_304', 'NV_305', 'NV_306', 'NV_307', 'NV_308', 'NV_309', 'NV_310',
#                   'NV_311', 'NV_312', 'NV_313', 'NV_314', 'NV_315', 'NV_316', 'NV_317', 'NV_318', 'NV_319', 'NV_320',
#                   'NV_321', 'NV_322', 'NV_323', 'NV_324', 'NV_325', 'NV_326', 'NV_327', 'NV_328', 'NV_329', 'NV_330',
#                   'NV_331', 'NV_332', 'NV_333', 'NV_334', 'NV_335', 'NV_336', 'NV_337', 'NV_338', 'NV_339', 'NV_340',
#                   'NV_341', 'NV_342', 'NV_343', 'NV_344', 'NV_345', 'NV_346', 'NV_347', 'NV_348', 'NV_349', 'NV_350',
#                   'NV_351', 'NV_352', 'NV_353', 'NV_354', 'NV_355', 'NV_356', 'NV_357', 'NV_358', 'NV_359', 'NV_360',
#                   'NV_361', 'NV_362', 'NV_363', 'NV_364', 'NV_365', 'NV_366', 'NV_367', 'NV_368', 'NV_369', 'NV_370',
#                   'NV_371', 'NV_372', 'NV_373', 'NV_374', 'NV_375', 'NV_376', 'NV_377', 'NV_378', 'NV_379', 'NV_380',
#                   'NV_381', 'NV_382', 'NV_383', 'NV_384', 'NV_385', 'NV_386', 'NV_387', 'NV_388', 'NV_389', 'NV_390',
#                   'NV_391', 'NV_392', 'NV_393', 'NV_394', 'NV_395', 'NV_396', 'NV_397', 'NV_398', 'NV_399', 'NV_400']
# no_tag = 'no_violence,'
# no_delete_more_new = []
# for e in no_delete_more:
#     no_delete_more_new.append(no_tag + e)
# print(no_delete_more_new)

# yes_delete_new = ['yes_violence,V_1', 'yes_violence,V_17', 'yes_violence,V_2', 'yes_violence,V_222',
#                   'yes_violence,V_236', 'yes_violence,V_237', 'yes_violence,V_272', 'yes_violence,V_276',
#                   'yes_violence,V_277', 'yes_violence,V_278', 'yes_violence,V_283', 'yes_violence,V_286',
#                   'yes_violence,V_291', 'yes_violence,V_294', 'yes_violence,V_298', 'yes_violence,V_299',
#                   'yes_violence,V_301', 'yes_violence,V_303', 'yes_violence,V_307', 'yes_violence,V_311',
#                   'yes_violence,V_313', 'yes_violence,V_315', 'yes_violence,V_320', 'yes_violence,V_332',
#                   'yes_violence,V_333', 'yes_violence,V_359', 'yes_violence,V_374', 'yes_violence,V_377',
#                   'yes_violence,V_387', 'yes_violence,V_398', 'yes_violence,V_407', 'yes_violence,V_411',
#                   'yes_violence,V_421', 'yes_violence,V_448', 'yes_violence,V_641', 'yes_violence,V_642',
#                   'yes_violence,V_643', 'yes_violence,V_644', 'yes_violence,V_645', 'yes_violence,V_646',
#                   'yes_violence,V_647', 'yes_violence,V_7', 'yes_violence,V_750', 'yes_violence,V_751',
#                   'yes_violence,V_759', 'yes_violence,V_760', 'yes_violence,V_761', 'yes_violence,V_789',
#                   'yes_violence,V_8', 'yes_violence,V_9']
# no_delete_new = ['no_violence,NV_1', 'no_violence,NV_11', 'no_violence,NV_14', 'no_violence,NV_17',
#                  'no_violence,NV_194', 'no_violence,NV_20', 'no_violence,NV_21', 'no_violence,NV_23',
#                  'no_violence,NV_25', 'no_violence,NV_26', 'no_violence,NV_269', 'no_violence,NV_27',
#                  'no_violence,NV_273', 'no_violence,NV_274', 'no_violence,NV_286', 'no_violence,NV_293',
#                  'no_violence,NV_295', 'no_violence,NV_296', 'no_violence,NV_297', 'no_violence,NV_298',
#                  'no_violence,NV_299', 'no_violence,NV_316', 'no_violence,NV_319', 'no_violence,NV_32',
#                  'no_violence,NV_320', 'no_violence,NV_322', 'no_violence,NV_353', 'no_violence,NV_354',
#                  'no_violence,NV_355', 'no_violence,NV_356', 'no_violence,NV_357', 'no_violence,NV_358',
#                  'no_violence,NV_359', 'no_violence,NV_360', 'no_violence,NV_361', 'no_violence,NV_362',
#                  'no_violence,NV_363', 'no_violence,NV_7', 'no_violence,NV_715', 'no_violence,NV_716',
#                  'no_violence,NV_721', 'no_violence,NV_761', 'no_violence,NV_823', 'no_violence,NV_824',
#                  'no_violence,NV_843', 'no_violence,NV_844', 'no_violence,NV_846', 'no_violence,NV_854',
#                  'no_violence,NV_974', 'no_violence,NV_992']
#
# # 修改train.csv、val.csv、test.csv(删除掉部分大的视频)
# with open("../splits/train_all.csv", 'r') as f1:
#     lines = f1.readlines()
#     with open("../splits/train.csv", 'w') as f2:
#         for line in lines:
#             line = line.strip()
#             if line in yes_delete_new or line in no_delete_new:
#                 continue
#             f2.write(line)
#             f2.write('\n')
#         f2.close()
#     f1.close()
# with open("../splits/val_all.csv", 'r') as f1:
#     lines = f1.readlines()
#     with open("../splits/val.csv", 'w') as f2:
#         for line in lines:
#             line = line.strip()
#             if line in yes_delete_new or line in no_delete_new:
#                 continue
#             f2.write(line)
#             f2.write('\n')
#         f2.close()
#     f1.close()
# with open("../splits/test_all.csv", 'r') as f1:
#     lines = f1.readlines()
#     with open("../splits/test.csv", 'w') as f2:
#         for line in lines:
#             line = line.strip()
#             if line in yes_delete_new or line in no_delete_new:
#                 continue
#             f2.write(line)
#             f2.write('\n')
#         f2.close()
#     f1.close()

no_delete_more_new = ['no_violence,NV_251', 'no_violence,NV_252', 'no_violence,NV_253', 'no_violence,NV_254',
                      'no_violence,NV_255', 'no_violence,NV_256', 'no_violence,NV_257', 'no_violence,NV_258',
                      'no_violence,NV_259', 'no_violence,NV_260', 'no_violence,NV_261', 'no_violence,NV_262',
                      'no_violence,NV_263', 'no_violence,NV_264', 'no_violence,NV_265', 'no_violence,NV_266',
                      'no_violence,NV_267', 'no_violence,NV_268', 'no_violence,NV_269', 'no_violence,NV_270',
                      'no_violence,NV_271', 'no_violence,NV_272', 'no_violence,NV_273', 'no_violence,NV_274',
                      'no_violence,NV_275', 'no_violence,NV_276', 'no_violence,NV_277', 'no_violence,NV_278',
                      'no_violence,NV_279', 'no_violence,NV_280', 'no_violence,NV_281', 'no_violence,NV_282',
                      'no_violence,NV_283', 'no_violence,NV_284', 'no_violence,NV_285', 'no_violence,NV_286',
                      'no_violence,NV_287', 'no_violence,NV_288', 'no_violence,NV_289', 'no_violence,NV_290',
                      'no_violence,NV_291', 'no_violence,NV_292', 'no_violence,NV_293', 'no_violence,NV_294',
                      'no_violence,NV_295', 'no_violence,NV_296', 'no_violence,NV_297', 'no_violence,NV_298',
                      'no_violence,NV_299', 'no_violence,NV_300', 'no_violence,NV_301', 'no_violence,NV_302',
                      'no_violence,NV_303', 'no_violence,NV_304', 'no_violence,NV_305', 'no_violence,NV_306',
                      'no_violence,NV_307', 'no_violence,NV_308', 'no_violence,NV_309', 'no_violence,NV_310',
                      'no_violence,NV_311', 'no_violence,NV_312', 'no_violence,NV_313', 'no_violence,NV_314',
                      'no_violence,NV_315', 'no_violence,NV_316', 'no_violence,NV_317', 'no_violence,NV_318',
                      'no_violence,NV_319', 'no_violence,NV_320', 'no_violence,NV_321', 'no_violence,NV_322',
                      'no_violence,NV_323', 'no_violence,NV_324', 'no_violence,NV_325', 'no_violence,NV_326',
                      'no_violence,NV_327', 'no_violence,NV_328', 'no_violence,NV_329', 'no_violence,NV_330',
                      'no_violence,NV_331', 'no_violence,NV_332', 'no_violence,NV_333', 'no_violence,NV_334',
                      'no_violence,NV_335', 'no_violence,NV_336', 'no_violence,NV_337', 'no_violence,NV_338',
                      'no_violence,NV_339', 'no_violence,NV_340', 'no_violence,NV_341', 'no_violence,NV_342',
                      'no_violence,NV_343', 'no_violence,NV_344', 'no_violence,NV_345', 'no_violence,NV_346',
                      'no_violence,NV_347', 'no_violence,NV_348', 'no_violence,NV_349', 'no_violence,NV_350',
                      'no_violence,NV_351', 'no_violence,NV_352', 'no_violence,NV_353', 'no_violence,NV_354',
                      'no_violence,NV_355', 'no_violence,NV_356', 'no_violence,NV_357', 'no_violence,NV_358',
                      'no_violence,NV_359', 'no_violence,NV_360', 'no_violence,NV_361', 'no_violence,NV_362',
                      'no_violence,NV_363', 'no_violence,NV_364', 'no_violence,NV_365', 'no_violence,NV_366',
                      'no_violence,NV_367', 'no_violence,NV_368', 'no_violence,NV_369', 'no_violence,NV_370',
                      'no_violence,NV_371', 'no_violence,NV_372', 'no_violence,NV_373', 'no_violence,NV_374',
                      'no_violence,NV_375', 'no_violence,NV_376', 'no_violence,NV_377', 'no_violence,NV_378',
                      'no_violence,NV_379', 'no_violence,NV_380', 'no_violence,NV_381', 'no_violence,NV_382',
                      'no_violence,NV_383', 'no_violence,NV_384', 'no_violence,NV_385', 'no_violence,NV_386',
                      'no_violence,NV_387', 'no_violence,NV_388', 'no_violence,NV_389', 'no_violence,NV_390',
                      'no_violence,NV_391', 'no_violence,NV_392', 'no_violence,NV_393', 'no_violence,NV_394',
                      'no_violence,NV_395', 'no_violence,NV_396', 'no_violence,NV_397', 'no_violence,NV_398',
                      'no_violence,NV_399', 'no_violence,NV_400']
# 修改train.csv、val.csv、test.csv(删除掉部分大的视频)
# with open("../splits/train_all.csv", 'r') as f1:
with open("../splits/train_with_no_violence251_400.csv", 'r') as f1:
    lines = f1.readlines()
    with open("../splits/train.csv", 'w') as f2:
        for line in lines:
            line = line.strip()
            # if line in yes_delete_new or line in no_delete_new:
            if line in no_delete_more_new:
                continue
            f2.write(line)
            f2.write('\n')
        f2.close()
    f1.close()
# with open("../splits/val_all.csv", 'r') as f1:
with open("../splits/val_with_no_violence251_400.csv", 'r') as f1:
    lines = f1.readlines()
    with open("../splits/val.csv", 'w') as f2:
        for line in lines:
            line = line.strip()
            # if line in yes_delete_new or line in no_delete_new:
            if line in no_delete_more_new:
                continue
            f2.write(line)
            f2.write('\n')
        f2.close()
    f1.close()
# with open("../splits/test_all.csv", 'r') as f1:
with open("../splits/test_with_no_violence251_400.csv", 'r') as f1:
    lines = f1.readlines()
    with open("../splits/test.csv", 'w') as f2:
        for line in lines:
            line = line.strip()
            # if line in yes_delete_new or line in no_delete_new:
            if line in no_delete_more_new:
                continue
            f2.write(line)
            f2.write('\n')
        f2.close()
    f1.close()
