from BScThesis import DicomImageAnalyser as dia

#
# experiments_data_folderpath = "TestImages/CleanUSScanCopy"
# dia.analizeDicom(experiments_data_folderpath, experiment_images=True)


# Patient Files
patient_data_excelfilepath = "ExcelInformation/Gynius-Pelvic_Floor_Disorders_excel_export_20191104115808.xlsx"
patient_data_folderpath_1 = "TestImages/Round 1"
patient_data_folderpath_3 = "TestImages/Round 3"

patient_data_for_filtering = [
    "TestImages/Round 2/130124_First_130124_First(5)_mod_rest",
    "TestImages/Round 2/130185_First_130185_First(5)_mod_rest",
    "TestImages/Round 2/130208_First_130208_First_mod_rest",
    "TestImages/Round 2/130328_First_130328_First_mod_rest",
    "TestImages/Round 2/130448_First_130448_First_mod_rest"
]

# Normal script
dia.analizeDicom(patient_data_folderpath_3, patient_data_excelfilepath, experiment_images=False)